#!/usr/bin/env python
"""Where do the supervised user-day labels have to come from?

Trains the two supervised baselines on the 75-dim user-day vectors with labels drawn from
different windows and evaluates every variant on the untouched test window. Produced because
the training-window rerun (scripts/userday.py, default mode) dropped the supervised user-day
AUC from 0.954 / 0.911 (labels from the validation window's first half) to 0.729 / 0.518,
and neither principal overlap nor scenario type explained it. Writes
results/userday/label_drift.csv; reuses the user-day caches userday.py leaves in --out.

    python scripts/userday_label_drift.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from userday import get_userdays, split_in_time  # noqa: E402

from sentinel.config import DATA_PROCESSED, RESULTS, SEEDS  # noqa: E402

warnings.filterwarnings("ignore")


def xy(*uds):
    return (np.vstack([np.asarray(u.x) for u in uds]),
            np.concatenate([np.asarray(u.y) for u in uds]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--out", type=Path, default=RESULTS / "userday")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = ap.parse_args(argv)

    tr = get_userdays(a.data / "train.parquet", a.out, "train")
    tm = get_userdays(a.data / "train_malicious.parquet", a.out, "train_malicious")
    val = get_userdays(a.data / "val.parquet", a.out, "val")
    test = get_userdays(a.data / "test.parquet", a.out, "test")
    v1, _ = split_in_time(val)
    xt, yt = xy(test)

    sources = {
        "training window (train + train_malicious)": (tr, tm),
        "validation window, first half by day": (v1,),
        "training window + validation first half": (tr, tm, v1),
        "validation window, whole": (val,),
    }
    rows = []
    for name, uds in sources.items():
        xs, ys = xy(*uds)
        for seed in a.seeds:
            lr = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed)).fit(xs, ys)
            rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1,
                                        random_state=seed).fit(xs, ys)
            rows.append({"label_source": name, "seed": seed, "n_userdays": len(ys),
                         "n_positive": int(ys.sum()),
                         "lr_test_auc": roc_auc_score(yt, lr.predict_proba(xt)[:, 1]),
                         "rf_test_auc": roc_auc_score(yt, rf.predict_proba(xt)[:, 1])})
            print(f"  {name:44s} seed {seed}  LR {rows[-1]['lr_test_auc']:.3f}  "
                  f"RF {rows[-1]['rf_test_auc']:.3f}", flush=True)
    per_seed = pd.DataFrame(rows)
    table = (per_seed.groupby("label_source", sort=False)
             .agg(n_userdays=("n_userdays", "first"), n_positive=("n_positive", "first"),
                  lr_test_auc=("lr_test_auc", "mean"), lr_test_auc_std=("lr_test_auc", "std"),
                  rf_test_auc=("rf_test_auc", "mean"), rf_test_auc_std=("rf_test_auc", "std"))
             .reset_index())
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / "label_drift.csv", index=False)
    per_seed.to_csv(a.out / "label_drift_per_seed.csv", index=False)
    (a.out / "label_drift_meta.json").write_text(json.dumps({
        "seeds": a.seeds, "evaluated_on": "test window, all user-days",
        "test_userdays": int(len(yt)), "test_positive": int(yt.sum()),
        "models": "StandardScaler + LogisticRegression(balanced) / RandomForest(200, balanced)",
    }, indent=2))
    print(table.round(3).to_string(index=False))
    print(f"wrote {a.out / 'label_drift.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
