#!/usr/bin/env python
"""Side-by-side Table V across engine variants -> results/table_v_variants.csv.

Reads ``table_v.csv`` from each variant directory under results/ and lays the models
out side by side, so the README can show the progression from the paper's global
design to per-source calibration and per-source models honestly.

    python scripts/compare_variants.py --results results \\
        --variant global --variant per_source_calibration --variant per_source_models
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.config import RESULTS  # noqa: E402

COLS = ("auc", "f1", "fpr_pct", "tuned_f1", "tuned_fpr_pct")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--variant", action="append", default=None)
    a = ap.parse_args(argv)
    variants = a.variant or [d.name for d in sorted(a.results.iterdir())
                             if (d / "table_v.csv").exists()]
    tables = {}
    for v in variants:
        path = a.results / v / "table_v.csv"
        if not path.exists():
            print(f"  (skipping {v}: no table_v.csv)")
            continue
        with path.open() as fh:
            tables[v] = {r["model"]: r for r in csv.DictReader(fh)}
    if not tables:
        raise SystemExit("no variant tables found")
    models = list(next(iter(tables.values())))
    header = ["model"] + [f"{v}:{c}" for v in tables for c in COLS]
    out = a.results / "table_v_variants.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for m in models:
            w.writerow([m] + [tables[v][m][c] for v in tables for c in COLS])
    print(f"{'model':22s}" + "".join(f"{v:>34s}" for v in tables))
    print(f"{'':22s}" + "".join(f"{'AUC':>8s}{'F1@.85':>8s}{'FPR%':>8s}{'tunedF1':>10s}"
                                for _ in tables))
    for m in models:
        print(f"{m:22s}" + "".join(f"{tables[v][m]['auc']:>8s}{tables[v][m]['f1']:>8s}"
                                   f"{tables[v][m]['fpr_pct']:>8s}{tables[v][m]['tuned_f1']:>10s}"
                                   for v in tables))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
