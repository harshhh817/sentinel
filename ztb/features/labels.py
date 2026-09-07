"""Attach malicious labels from the CERT ``answers/`` tree.

The r4.2 release marks scripted insider activity in ``answers/``, whose layout varies
between mirrors: some ship a flat set of ``r4.2-N-USER-*.csv`` files, others nest them
under per-scenario directories, and most include an ``insiders.csv`` roster. Rather
than hard-coding one layout, this module walks the tree and extracts CERT event ids
wherever they appear, which is stable across all of them: an event id is a brace-
delimited token such as ``{H3P8-M1SE43GX-4444RZOJ}``.

Labels are event-level. The paper reports 3,912 malicious events (0.179 %) in the
five-month test window; :func:`label_report` exists so that figure can be checked
against whatever the corpus actually yields rather than assumed.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# CERT event ids are brace-delimited, uppercase alphanumeric with hyphens.
EVENT_ID_RE = re.compile(r"\{[A-Z0-9]{4}-[A-Z0-9]{8}-[A-Z0-9]{8}\}")

# Looser fallback: any brace-delimited token. Used only if the strict form matches
# nothing, so a differently-formatted release still produces labels.
LOOSE_ID_RE = re.compile(r"\{[A-Za-z0-9\-]{6,}\}")


@dataclass(frozen=True)
class LabelSet:
    """Malicious event ids and the insider principals they belong to."""

    event_ids: frozenset[str]
    insiders: frozenset[str]
    scenarios: dict[str, int]  # scenario name -> number of events contributed

    def __contains__(self, event_id: str) -> bool:
        return event_id in self.event_ids

    def __len__(self) -> int:
        return len(self.event_ids)


def _scenario_name(path: Path, root: Path) -> str:
    """``answers/r4.2-2/r4.2-2-CDE1846-...csv`` -> ``r4.2-2``."""
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = rel.stem
    parts = stem.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else stem


def load_labels(answers_dir: Path) -> LabelSet:
    """Walk ``answers/`` and collect every malicious event id it references."""
    if not answers_dir.is_dir():
        return LabelSet(frozenset(), frozenset(), {})

    event_ids: set[str] = set()
    scenarios: dict[str, int] = {}
    insiders: set[str] = set()

    for path in sorted(answers_dir.rglob("*.csv")):
        if path.name.lower() == "insiders.csv":
            insiders |= _read_insiders(path)
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        found = set(EVENT_ID_RE.findall(text))
        if not found:
            found = set(LOOSE_ID_RE.findall(text))
        if not found:
            continue

        scenario = _scenario_name(path, answers_dir)
        new = found - event_ids
        scenarios[scenario] = scenarios.get(scenario, 0) + len(new)
        event_ids |= found

    return LabelSet(frozenset(event_ids), frozenset(insiders), scenarios)


def _read_insiders(path: Path) -> set[str]:
    """Pull user ids out of ``insiders.csv`` whatever its column order."""
    users: set[str] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return users
        cols = [h.strip().lower() for h in header]
        try:
            idx = next(i for i, c in enumerate(cols) if c in {"user", "user_id", "userid"})
        except StopIteration:
            return users
        for row in reader:
            if len(row) > idx and row[idx].strip():
                users.add(row[idx].strip())
    return users


def label_report(labels: LabelSet, total_events: int, positives: int) -> dict[str, object]:
    """Summary for the module report — never hand-type a positive rate."""
    return {
        "malicious_event_ids_in_answers": len(labels),
        "insider_principals": len(labels.insiders),
        "scenarios": dict(sorted(labels.scenarios.items())),
        "events_seen": total_events,
        "positives_matched": positives,
        "positive_rate_pct": round(100.0 * positives / total_events, 4) if total_events else 0.0,
    }


def apply_labels(events: Iterable, labels: LabelSet):
    """Set ``label`` on a stream of :class:`~ztb.features.schema.CloudEvent`."""
    for event in events:
        event.label = 1 if event.event_id in labels.event_ids else 0
        yield event
