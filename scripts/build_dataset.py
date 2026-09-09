#!/usr/bin/env python
"""Replay the CERT corpus in time order and write the train/val/test splits.

Streams events through the mapper, builds each 34-dim vector against the principal's
baseline *as of that moment*, folds the event into the baseline, and appends the row to
whichever split its timestamp falls in. Nothing is held in memory beyond one batch, so
this runs on a laptop against a 10 GB ``http.csv``.

The split is **by time**, per Section V: the first 10 months train (benign only), the
next 2 validate, the final 5 test. A random split would let a model see a principal's
future while scoring their past.

    python scripts/build_dataset.py --root data/raw/r4.2              # full 17 months
    python scripts/build_dataset.py --root data/raw/r4.2 --months 1   # 1-month sample

The one-month sample writes to ``data/processed/sample/`` and exists to validate the
mapping and feature builder before committing to a full pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import DATA_PROCESSED, DATA_RAW, MODELS, RESULTS, SPLIT_MONTHS  # noqa: E402
from ztb.features.baseline import BaselineStore  # noqa: E402
from ztb.features.builder import (  # noqa: E402
    FEATURE_NAMES,
    StandardiserAccumulator,
    build_vector,
    observe_event,
)
from ztb.features.cert_mapper import load_org_units, stream_events  # noqa: E402
from ztb.features.labels import label_report, load_labels  # noqa: E402

BATCH_ROWS = 10_000
# Average days per month, used to place the split boundaries from the corpus start.
DAYS_PER_MONTH = 30.44


@dataclass
class SplitBoundaries:
    """Timestamps separating train | val | test."""

    start: datetime
    train_end: datetime
    val_end: datetime

    def of(self, ts: datetime) -> str:
        if ts < self.train_end:
            return "train"
        if ts < self.val_end:
            return "val"
        return "test"

    def to_dict(self) -> dict[str, str]:
        return {
            "corpus_start": self.start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_end": self.val_end.isoformat(),
        }


def corpus_start(root: Path) -> datetime:
    """Earliest timestamp across the mapped sources, read from their first rows."""
    first = None
    for event in stream_events(root):
        first = event.ts
        break
    if first is None:
        raise SystemExit(f"no events found under {root}")
    return first


def boundaries_from(start: datetime) -> SplitBoundaries:
    train_end = start + timedelta(days=SPLIT_MONTHS["train"] * DAYS_PER_MONTH)
    val_end = train_end + timedelta(days=SPLIT_MONTHS["val"] * DAYS_PER_MONTH)
    return SplitBoundaries(start=start, train_end=train_end, val_end=val_end)


class SplitWriter:
    """Buffered parquet writer, one per split, flushing every ``BATCH_ROWS`` rows."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self.path = path
        self.schema = schema
        self._writer: pq.ParquetWriter | None = None
        self._rows: list[dict] = []
        self.count = 0
        self.positives = 0

    def add(self, row: dict) -> None:
        self._rows.append(row)
        self.count += 1
        self.positives += row["label"]
        if len(self._rows) >= BATCH_ROWS:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="snappy")
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is None:
            # Always leave a (possibly empty) file so downstream scripts can rely on it.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(self.schema.empty_table(), self.path, compression="snappy")
            return
        self._writer.close()


def arrow_schema() -> pa.Schema:
    fields = [
        pa.field("event_id", pa.string()),
        pa.field("ts", pa.timestamp("us")),
        pa.field("principal", pa.string()),
        pa.field("action", pa.string()),
        pa.field("resource", pa.string()),
        pa.field("source", pa.string()),
        pa.field("label", pa.int8()),
    ]
    fields += [pa.field(name, pa.float64()) for name in FEATURE_NAMES]
    return pa.schema(fields)


def build(
    root: Path,
    out_dir: Path,
    *,
    months: int | None = None,
    limit: int | None = None,
    progress_every: int = 250_000,
) -> dict:
    """Stream the corpus into ``out_dir`` and return a report dict."""
    labels = load_labels(root / "answers")
    org = load_org_units(root / "LDAP")
    print(f"answers/: {len(labels)} malicious event ids, "
          f"{len(labels.insiders)} insider principals")
    print(f"LDAP:     {len(org)} principals")

    start = corpus_start(root)
    if months is not None:
        # Sample mode: one contiguous window from the corpus start, split proportionally
        # so the sample exercises all three branches.
        until = start + timedelta(days=months * DAYS_PER_MONTH)
        total_months = sum(SPLIT_MONTHS.values())
        scale = months / total_months
        bounds = SplitBoundaries(
            start=start,
            train_end=start + timedelta(days=SPLIT_MONTHS["train"] * DAYS_PER_MONTH * scale),
            val_end=start + timedelta(
                days=(SPLIT_MONTHS["train"] + SPLIT_MONTHS["val"]) * DAYS_PER_MONTH * scale
            ),
        )
    else:
        until = None
        bounds = boundaries_from(start)

    print(f"corpus starts {start.isoformat()}")
    print(f"split: train < {bounds.train_end.isoformat()} "
          f"<= val < {bounds.val_end.isoformat()} <= test")

    schema = arrow_schema()
    # train_malicious holds the scripted-scenario rows that fall inside the training
    # window. The unsupervised detectors never see them (Section V); the supervised
    # baselines of Table V do, which is what makes them "trained with malicious
    # labels available".
    writers = {name: SplitWriter(out_dir / f"{name}.parquet", schema)
               for name in ("train", "train_malicious", "val", "test")}
    store = BaselineStore()
    # Welford accumulator, not a list of vectors: the training window does not fit in
    # memory on the full corpus.
    train_stats = StandardiserAccumulator()
    seen = 0
    train_malicious_dropped = 0
    by_source: dict[str, int] = {}
    by_action: dict[str, int] = {}

    for event in stream_events(root, until=until, labels=set(labels.keys)):
        profile = store.get(event.principal)
        vector = build_vector(event, profile, org.get(event.principal))
        observe_event(event, profile)

        seen += 1
        by_source[event.source] = by_source.get(event.source, 0) + 1
        by_action[event.action] = by_action.get(event.action, 0) + 1

        split = bounds.of(event.ts)
        # The training window is benign-only (Section V): both detectors are
        # unsupervised and must never see a scripted scenario. Those rows go to a
        # side file for the supervised baselines instead of being discarded.
        if split == "train" and event.label == 1:
            train_malicious_dropped += 1
            split = "train_malicious"

        row = {
            "event_id": event.event_id,
            "ts": event.ts,
            "principal": event.principal,
            "action": event.action,
            "resource": event.resource,
            "source": event.source,
            "label": event.label,
        }
        row.update(dict(zip(FEATURE_NAMES, vector.tolist(), strict=True)))
        writers[split].add(row)

        if split == "train":
            train_stats.update(vector)

        if progress_every and seen % progress_every == 0:
            print(f"  {seen:,} events mapped  "
                  f"(train={writers['train'].count:,} val={writers['val'].count:,} "
                  f"test={writers['test'].count:,})", flush=True)

        if limit is not None and seen >= limit:
            break

    for writer in writers.values():
        writer.close()

    # Standardisation statistics are fitted on the training window ONLY.
    standardiser = None
    if len(train_stats):
        standardiser = train_stats.finalize()
        MODELS.mkdir(parents=True, exist_ok=True)
        (MODELS / "standardiser.json").write_text(json.dumps(standardiser.to_dict(), indent=2))

    positives = sum(w.positives for w in writers.values())
    report = {
        "root": str(root),
        "out_dir": str(out_dir),
        "months_sampled": months,
        "boundaries": bounds.to_dict(),
        "events_mapped": seen,
        "events_by_source": dict(sorted(by_source.items())),
        "events_by_action": dict(sorted(by_action.items())),
        "principals": len(store),
        "train_malicious_dropped": train_malicious_dropped,
        "splits": {
            name: {
                "rows": w.count,
                "positives": w.positives,
                "positive_rate_pct": round(100.0 * w.positives / w.count, 4) if w.count else 0.0,
            }
            for name, w in writers.items()
        },
        "labels": label_report(labels, seen, positives),
        "standardiser_fitted_on": "train" if standardiser else None,
        "standardiser_rows": len(train_stats),
    }
    payload = json.dumps(report, indent=2, default=str)
    (out_dir / "report.json").write_text(payload)
    # data/processed lives on an external disk and is gitignored; results/ is the
    # committed record, and CLAUDE.md requires every number there to come from a script.
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "module1_sample_report.json" if months else "module1_dataset_report.json"
    (RESULTS / name).write_text(payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_RAW / "r4.2",
                        help="directory holding logon.csv, file.csv, device.csv, http.csv, "
                             "LDAP/ and answers/")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--months", type=int, default=None,
                        help="validate on a contiguous N-month sample instead of the full corpus")
    parser.add_argument("--limit", type=int, default=None, help="stop after N events")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(
            f"{args.root} does not exist. Place the extracted CERT r4.2 CSVs there "
            "(logon.csv, file.csv, device.csv, http.csv, LDAP/, answers/)."
        )

    out = args.out or (DATA_PROCESSED / "sample" if args.months else DATA_PROCESSED)
    out.mkdir(parents=True, exist_ok=True)
    report = build(args.root, out, months=args.months, limit=args.limit)

    print("\n--- report ---")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
