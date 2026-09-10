#!/usr/bin/env python
"""Detector AUC broken down by CERT event source -> --out/module2_auc_by_source.csv.

Explains why a single global manifold + global calibration under-performs on this
corpus: 94 % of events are http, so the benign calibration CDFs are dominated by
http rows and every logon/device/file event lands near the top of the benign
quantile regardless of intent.

    python scripts/diagnose_by_source.py --data data/processed --models models --split val
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import DATA_PROCESSED, MODELS, RESULTS  # noqa: E402
from ztb.risk.data import load_split  # noqa: E402
from ztb.risk.fusion import RiskEngine  # noqa: E402

SOURCES = ("logon", "device", "file", "http")


def _auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _mean(v):
    return round(float(v.mean()), 4) if len(v) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args(argv)

    path = a.data / f"{a.split}.parquet"
    split = load_split(path, max_rows=a.max_rows, seed=0)
    src = pq.read_table(path, columns=["source"]).column("source").to_numpy(zero_copy_only=False)
    if a.max_rows and len(src) != len(split):
        raise SystemExit("--max-rows needs the full split to align the source column")
    sc = RiskEngine.load(a.models, a.seed).score(split.x)

    rows = []
    groups = [(s, src == s) for s in SOURCES]
    groups += [("non-http", src != "http"), ("all", np.ones(len(src), bool))]
    for name, m in groups:
        y = split.y[m]
        r = sc.r[m]
        rows.append({
            "split": a.split, "seed": a.seed, "source": name, "rows": int(m.sum()),
            "positives": int(y.sum()),
            "auc_hybrid": round(_auc(y, r), 4),
            "auc_autoencoder": round(_auc(y, sc.f_e[m]), 4),
            "auc_iforest": round(_auc(y, sc.f_s[m]), 4),
            "r_benign_mean": _mean(r[y == 0]),
            "r_malicious_mean": _mean(r[y == 1]),
        })
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "module2_auc_by_source.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['source']:9s} rows {r['rows']:>10,} pos {r['positives']:>5}  "
              f"AUC hybrid {r['auc_hybrid']:.3f}  AE {r['auc_autoencoder']:.3f}  "
              f"iF {r['auc_iforest']:.3f}  r benign {r['r_benign_mean']:.3f} / "
              f"mal {r['r_malicious_mean']:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
