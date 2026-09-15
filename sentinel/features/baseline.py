"""Per-principal behavioural baseline (Section IV-B).

Everything the feature builder needs about a principal's *past* lives here: how often
they call each action, which resources and hosts they have touched, their modal working
hours, and the sliding windows behind the rate features.

Two rules from the paper drive the design:

* **EWMA with a 30-day effective half-life.** Counts decay continuously toward the
  request timestamp rather than being reset in buckets, so the profile adapts to a
  genuine role change while bounding how fast a patient insider (A2) can drag their own
  baseline. Decay is applied lazily on read/write, so cost is O(1) per event and no
  sweep over idle principals is ever needed.
* **Sliding windows anchored on the request timestamp**, not wall-clock buckets, so an
  adversary cannot straddle a bucket boundary to hide a burst.

The store is deliberately update-in-place and single-threaded: ``scripts/build_dataset``
replays events in timestamp order, so a principal's profile at the moment an event is
scored contains that principal's history and nothing from its future.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from sentinel.config import BASELINE_HALF_LIFE_DAYS

# Longest sliding window the rate features need; older entries are discarded.
MAX_WINDOW_SECONDS = 3600


def _decay_factor(seconds: float, half_life_days: float = BASELINE_HALF_LIFE_DAYS) -> float:
    """Weight an observation ``seconds`` old under the configured half-life."""
    if seconds <= 0:
        return 1.0
    return 0.5 ** (seconds / (half_life_days * 86400.0))


@dataclass
class PrincipalProfile:
    """One principal's decayed history. All counts are EWMA weights, not raw counts."""

    principal: str
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    # Decayed counts, all anchored at ``decay_anchor``.
    decay_anchor: datetime | None = None
    total: float = 0.0
    action_counts: dict[str, float] = field(default_factory=dict)
    resource_counts: dict[str, float] = field(default_factory=dict)
    pc_counts: dict[str, float] = field(default_factory=dict)
    asn_counts: dict[str, float] = field(default_factory=dict)
    hour_counts: list[float] = field(default_factory=lambda: [0.0] * 24)
    reads: float = 0.0
    writes: float = 0.0

    # Exact, undecayed history used for novelty flags ("has this ever happened?").
    seen_actions: set[str] = field(default_factory=set)
    seen_resources: set[str] = field(default_factory=set)
    seen_pcs: set[str] = field(default_factory=set)
    seen_asns: set[str] = field(default_factory=set)

    # Sliding windows: (timestamp, resource, action, bytes_read) within the last hour.
    window: deque = field(default_factory=deque)

    # State carried between consecutive requests.
    prev_ts: datetime | None = None
    prev_location: tuple[float, float] | None = None
    last_logon_ts: datetime | None = None

    # --- decay -------------------------------------------------------------

    def _decay_to(self, ts: datetime) -> None:
        """Age every decayed count forward to ``ts``. Called before any read or write."""
        if self.decay_anchor is None:
            self.decay_anchor = ts
            return
        elapsed = (ts - self.decay_anchor).total_seconds()
        if elapsed <= 0:
            return
        factor = _decay_factor(elapsed)
        self.decay_anchor = ts
        if factor >= 1.0:
            return
        self.total *= factor
        self.reads *= factor
        self.writes *= factor
        self.hour_counts = [c * factor for c in self.hour_counts]
        for counts in (self.action_counts, self.resource_counts,
                       self.pc_counts, self.asn_counts):
            for key in list(counts):
                decayed = counts[key] * factor
                # Drop negligible entries so high-cardinality dicts stay bounded.
                if decayed < 1e-6:
                    del counts[key]
                else:
                    counts[key] = decayed

    def _trim_window(self, ts: datetime) -> None:
        cutoff = ts.timestamp() - MAX_WINDOW_SECONDS
        while self.window and self.window[0][0] < cutoff:
            self.window.popleft()

    # --- queries (read-only; must be called before :meth:`observe`) ---------

    def action_frequency(self, action: str, ts: datetime) -> float:
        """Empirical frequency of ``action`` in this principal's own history."""
        self._decay_to(ts)
        if self.total <= 0:
            return 0.0
        return self.action_counts.get(action, 0.0) / self.total

    def resource_frequency(self, resource: str, ts: datetime) -> float:
        self._decay_to(ts)
        if self.total <= 0:
            return 0.0
        return self.resource_counts.get(resource, 0.0) / self.total

    def action_entropy(self, ts: datetime) -> float:
        """Shannon entropy of the decayed action distribution, in nats."""
        self._decay_to(ts)
        if self.total <= 0:
            return 0.0
        entropy = 0.0
        for count in self.action_counts.values():
            p = count / self.total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def read_write_ratio(self, ts: datetime) -> float:
        self._decay_to(ts)
        denominator = self.reads + self.writes
        return self.reads / denominator if denominator > 0 else 0.5

    def modal_hour_deviation(self, ts: datetime) -> float:
        """Circular distance from the principal's modal hour, normalised to [0, 1]."""
        self._decay_to(ts)
        if self.total <= 0:
            return 0.0
        modal = max(range(24), key=lambda h: self.hour_counts[h])
        diff = abs(ts.hour - modal)
        return min(diff, 24 - diff) / 12.0

    def window_counts(self, ts: datetime) -> tuple[int, int, int, int, int]:
        """``(calls_1m, calls_15m, calls_60m, distinct_resources_60m, distinct_actions_60m)``.

        Windows are anchored on ``ts`` itself, per Section IV-B.
        """
        self._trim_window(ts)
        now = ts.timestamp()
        c1 = c15 = c60 = 0
        resources: set[str] = set()
        actions: set[str] = set()
        for entry_ts, resource, action, _ in self.window:
            age = now - entry_ts
            if age <= 60:
                c1 += 1
            if age <= 900:
                c15 += 1
            if age <= 3600:
                c60 += 1
                resources.add(resource)
                actions.add(action)
        return c1, c15, c60, len(resources), len(actions)

    def bytes_read_60m(self, ts: datetime) -> float:
        self._trim_window(ts)
        cutoff = ts.timestamp() - 3600
        return float(sum(b for t, _, _, b in self.window if t >= cutoff))

    def inter_request_gap(self, ts: datetime) -> float:
        """Seconds since this principal's previous request; -1 for the first one."""
        if self.prev_ts is None:
            return -1.0
        return max(0.0, (ts - self.prev_ts).total_seconds())

    def tenure_days(self, ts: datetime) -> float:
        if self.first_seen is None:
            return 0.0
        return max(0.0, (ts - self.first_seen).total_seconds() / 86400.0)

    def mfa_age_seconds(self, ts: datetime) -> float:
        """Time since the last logon, standing in for the age of the MFA assertion."""
        if self.last_logon_ts is None:
            return -1.0
        return max(0.0, (ts - self.last_logon_ts).total_seconds())

    # --- update ------------------------------------------------------------

    def observe(
        self,
        ts: datetime,
        action: str,
        resource: str,
        pc: str,
        asn: str,
        *,
        is_read: bool,
        bytes_read: float = 0.0,
        location: tuple[float, float] | None = None,
        is_logon: bool = False,
    ) -> None:
        """Fold one event into the profile. Call *after* building its feature vector."""
        self._decay_to(ts)

        if self.first_seen is None:
            self.first_seen = ts
        self.last_seen = ts

        self.total += 1.0
        self.action_counts[action] = self.action_counts.get(action, 0.0) + 1.0
        self.resource_counts[resource] = self.resource_counts.get(resource, 0.0) + 1.0
        if pc:
            self.pc_counts[pc] = self.pc_counts.get(pc, 0.0) + 1.0
            self.seen_pcs.add(pc)
        if asn:
            self.asn_counts[asn] = self.asn_counts.get(asn, 0.0) + 1.0
            self.seen_asns.add(asn)
        self.hour_counts[ts.hour] += 1.0
        if is_read:
            self.reads += 1.0
        else:
            self.writes += 1.0

        self.seen_actions.add(action)
        self.seen_resources.add(resource)

        self._trim_window(ts)
        self.window.append((ts.timestamp(), resource, action, bytes_read))

        self.prev_ts = ts
        if location is not None:
            self.prev_location = location
        if is_logon:
            self.last_logon_ts = ts


class BaselineStore:
    """Per-principal profiles. Dict-backed locally; DynamoDB in cloud mode."""

    def __init__(self) -> None:
        self._profiles: dict[str, PrincipalProfile] = {}

    def get(self, principal: str) -> PrincipalProfile:
        profile = self._profiles.get(principal)
        if profile is None:
            profile = PrincipalProfile(principal=principal)
            self._profiles[principal] = profile
        return profile

    def __len__(self) -> int:
        return len(self._profiles)

    def __contains__(self, principal: str) -> bool:
        return principal in self._profiles

    def principals(self) -> list[str]:
        return sorted(self._profiles)
