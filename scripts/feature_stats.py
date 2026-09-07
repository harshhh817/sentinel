#!/usr/bin/env python
"""Per-feature summary statistics for the Module 1 report -> results/module1_feature_stats.csv.

Reads the parquet splits in chunks (row groups), so it does not load the training
window into memory. Reports, per feature and split: mean, std, min, max, and the
fraction of zeros (which exposes degenerate flags such as a rule that never fires).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import DATA_PROCESSED, RESULTS  # noqa: E402
from ztb.features.builder import FEATURE_NAMES  # noqa: E402


def summarise(path: Path) -> dict[str, dict[str, float]]:
    n = 0
    s = np.zeros(len(FEATURE_NAMES))
    s2 = np.zeros(len(FEATURE_NAMES))
    lo = np.full(len(FEATURE_NAMES), np.inf)
    hi = np.full(len(FEATURE_NAMES), -np.inf)
    zeros = np.zeros(len(FEATURE_NAMES))
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        m = pf.read_row_group(rg, columns=list(FEATURE_NAMES)).to_pandas().to_numpy(np.float64)
        n += len(m)
        s += m.sum(0)
        s2 += (m * m).sum(0)
        lo = np.minimum(lo, m.min(0))
        hi = np.maximum(hi, m.max(0))
        zeros += (m == 0).sum(0)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 0.0))
    return {
        name: {"n": n, "mean": mean[i], "std": std[i], "min": lo[i], "max": hi[i],
               "zero_frac": zeros[i] / n}
        for i, name in enumerate(FEATURE_NAMES)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--out", type=Path, default=RESULTS / "module1_feature_stats.csv")
    a = ap.parse_args()

    rows = []
    for split in ("train", "val", "test"):
        path = a.inp / f"{split}.parquet"
        if not path.exists():
            continue
        for name, st in summarise(path).items():
            rows.append({"split": split, "feature": name,
                         **{k: (v if k == "n" else round(float(v), 6)) for k, v in st.items()}})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
