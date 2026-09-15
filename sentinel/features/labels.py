"""Attach malicious labels from the CERT ``answers/`` tree.

The r4.2 release marks scripted insider activity in ``answers/``, whose layout varies
between mirrors: some ship a flat set of ``r4.2-N-USER-*.csv`` files, others nest them
under per-scenario directories, and most include an ``insiders.csv`` roster. Rather
than hard-coding one layout, this module walks the tree and extracts CERT event ids
wherever they appear, which is stable across all of them: an event id is a brace-
delimited token such as ``{H3P8-M1SE43GX-4444RZOJ}``.

The ``answers/`` archive on KiltHub is shared across every CERT release (r2 through
r6.2) and ``insiders.csv`` lists all of them with a bare ``dataset`` column such as
``4.2``. :func:`load_labels` therefore takes a ``release`` and only reads scenario
files whose name starts with ``r<release>-`` and insider rows whose dataset matches,
so dropping the whole shared tree into ``data/raw/r4.2/answers/`` is safe.

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
    """Malicious events, keyed on ``(source, event_id)``, and the insiders involved.

    The r4.2 readme's erratum: *"Field Ids are unique within a csv file but may not be
    globally unique."* An id alone can therefore name a benign row in one file and a
    malicious row in another, so the key carries the originating file as well.
    """

    keys: frozenset[tuple[str, str]]      # (source, event_id)
    insiders: frozenset[str]
    scenarios: dict[str, int]              # scenario name -> events contributed

    @property
    def event_ids(self) -> frozenset[str]:
        return frozenset(eid for _, eid in self.keys)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self.keys

    def __len__(self) -> int:
        return len(self.keys)


def _scenario_name(path: Path, root: Path) -> str:
    """``answers/r4.2-2/r4.2-2-CDE1846-...csv`` -> ``r4.2-2``."""
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = rel.stem
    parts = stem.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else stem


def load_labels(answers_dir: Path, release: str = "4.2") -> LabelSet:
    """Walk ``answers/`` and collect every malicious event id for ``release``."""
    if not answers_dir.is_dir():
        return LabelSet(frozenset(), frozenset(), {})

    keys: set[tuple[str, str]] = set()
    scenarios: dict[str, int] = {}
    insiders: set[str] = set()
    prefix = f"r{release}-"

    for path in sorted(answers_dir.rglob("*.csv")):
        if path.name.lower() == "insiders.csv":
            insiders |= _read_insiders(path, release)
            continue
        # The shared tree also holds r4.1-*.csv, r5.2-*/ and so on; skip them.
        if not path.name.startswith(prefix):
            continue

        found = _read_answer_rows(path)
        if not found:
            continue
        scenario = _scenario_name(path, answers_dir)
        scenarios[scenario] = scenarios.get(scenario, 0) + len(found - keys)
        keys |= found

    return LabelSet(frozenset(keys), frozenset(insiders), scenarios)


def _read_answer_rows(path: Path) -> set[tuple[str, str]]:
    """Answer rows are headerless: ``source,{id},date,user,pc,...``.

    Falls back to a regex sweep with an empty source if a row is not in that shape,
    so an unexpected layout still yields ids rather than silently nothing.
    """
    found: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2 and EVENT_ID_RE.fullmatch(row[1].strip()):
                found.add((row[0].strip().lower(), row[1].strip()))
    if not found:
        text = path.read_text(encoding="utf-8", errors="replace")
        ids = EVENT_ID_RE.findall(text) or LOOSE_ID_RE.findall(text)
        found = {("", eid) for eid in ids}
    return found


def _read_insiders(path: Path, release: str) -> set[str]:
    """Pull user ids for one release out of ``insiders.csv``.

    The real file's ``dataset`` column is bare (``4.2``); older mirrors used ``r4.2``.
    Both are accepted. A file with no ``dataset`` column is taken to be single-release.
    """
    users: set[str] = set()
    accepted = {release, f"r{release}"}
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return users
        cols = [h.strip().lower() for h in header]
        try:
            user_idx = next(i for i, c in enumerate(cols) if c in {"user", "user_id", "userid"})
        except StopIteration:
            return users
        dataset_idx = cols.index("dataset") if "dataset" in cols else None
        for row in reader:
            if len(row) <= user_idx or not row[user_idx].strip():
                continue
            if dataset_idx is not None and row[dataset_idx].strip() not in accepted:
                continue
            users.add(row[user_idx].strip())
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
    """Set ``label`` on a stream of :class:`~sentinel.features.schema.CloudEvent`."""
    for event in events:
        event.label = 1 if (event.source, event.event_id) in labels else 0
        yield event
