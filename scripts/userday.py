#!/usr/bin/env python
"""User-day granularity: aggregate, train, score, and propagate -> --out/.

1. Aggregate each split into (principal, day) vectors (ztb.risk.userday).
2. Train the same autoencoder + isolation forest on benign user-days of the training
   window; calibrate on benign user-days of the calibration window; fuse with alpha.
3. Table V at user-day granularity (a user-day is positive if any event is).
4. Propagate: each request's r becomes max(request-level r, its principal's current-day
   r); Table V at request granularity with the propagated score.

Windows: by default train = train.parquet, calibrate = val.parquet. When the training
window is unavailable (``--train-from-val``), the validation window is split in time:
its first half trains, its second half calibrates; test is untouched either way.

    python scripts/userday.py --data <splits> --models models --out results/userday
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import LABELS, MODEL_ORDER, _write_table_v, risk_scores  # noqa: E402

from ztb.config import DATA_PROCESSED, FUSION_ALPHA, MODELS, RESULTS, SEEDS  # noqa: E402
from ztb.risk.autoencoder import AEConfig, train_autoencoder  # noqa: E402
from ztb.risk.baselines import SUPERVISED, baseline_risk, fit_baseline  # noqa: E402
from ztb.risk.data import load_split  # noqa: E402
from ztb.risk.fusion import EmpiricalCDF, RiskEngine, load_engine  # noqa: E402
from ztb.risk.iforest import isolation_score, train_iforest  # noqa: E402
from ztb.risk.metrics import at_threshold, best_f1_threshold, summarise  # noqa: E402
from ztb.risk.trust import DENY_THRESHOLD, effective_risk  # noqa: E402
from ztb.risk.userday import (  # noqa: E402
    USERDAY_FEATURES,
    UserDays,
    aggregate,
    fit_standardiser,
    load,
    save,
)


def get_userdays(split_path: Path, cache_dir: Path, name: str) -> UserDays:
    cached = cache_dir / f"userday_{name}.parquet"
    if cached.exists():
        return load(cached)
    print(f"  aggregating {name} ...", flush=True)
    ud = aggregate(split_path, progress=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save(ud, cached)
    print(f"  {name}: {len(ud):,} user-days, {int(ud.y.sum())} positive", flush=True)
    return ud


def split_in_time(ud: UserDays) -> tuple[UserDays, UserDays]:
    days = np.sort(np.unique(ud.day))
    cut = days[len(days) // 2]
    return ud.subset(ud.day < cut), ud.subset(ud.day >= cut)


def train_userday_engine(tr: UserDays, cal: UserDays, *, seed: int, epochs: int, device: str,
                         alpha: float) -> RiskEngine:
    std = fit_standardiser(tr.benign.x)
    x_tr = std.transform(tr.benign.x.astype(np.float64)).astype(np.float32)
    x_cal = std.transform(cal.benign.x.astype(np.float64)).astype(np.float32)
    cfg = AEConfig(widths=(x_tr.shape[1], 24, 16, 8), epochs=epochs)
    ae, _ = train_autoencoder(x_tr, x_cal, config=cfg, seed=seed, device=device)
    forest = train_iforest(x_tr, seed=seed)
    e_cal, s_cal = ae.reconstruction_error(x_cal), isolation_score(forest, x_cal)
    return RiskEngine(std, ae, forest, EmpiricalCDF(e_cal), EmpiricalCDF(s_cal), alpha=alpha,
                      feature_names=USERDAY_FEATURES)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--models", type=Path, default=MODELS,
                    help="request-level engines used for the propagation baseline")
    ap.add_argument("--out", type=Path, default=RESULTS / "userday")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=AEConfig().epochs)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-from-val", action="store_true")
    ap.add_argument("--eval-subsample", type=int, default=4_000_000)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    # 1. user-days
    if a.train_from_val:
        val_all = get_userdays(a.data / "val.parquet", a.out, "val")
        tr_ud, cal_ud = split_in_time(val_all)
        windows = {"train": "val (first half, by day)", "calibrate": "val (second half, by day)"}
    else:
        tr_ud = get_userdays(a.data / "train.parquet", a.out, "train")
        tm = a.data / "train_malicious.parquet"
        if tm.exists():
            m_ud = get_userdays(tm, a.out, "train_malicious")
            tr_ud = UserDays(np.r_[tr_ud.principal, m_ud.principal], np.r_[tr_ud.day, m_ud.day],
                             np.vstack([tr_ud.x, m_ud.x]), np.r_[tr_ud.y, m_ud.y],
                             np.r_[tr_ud.n_events, m_ud.n_events])
        cal_ud = get_userdays(a.data / "val.parquet", a.out, "val")
        windows = {"train": "train (+ train_malicious for supervised)", "calibrate": "val"}
    te_ud = get_userdays(a.data / "test.parquet", a.out, "test")
    print(f"user-days: train {len(tr_ud):,} ({int(tr_ud.y.sum())} pos)  "
          f"calibrate {len(cal_ud):,} ({int(cal_ud.y.sum())} pos)  "
          f"test {len(te_ud):,} ({int(te_ud.y.sum())} pos)", flush=True)

    # 2-3. user-day Table V
    per_seed: dict[str, list[dict]] = {m: [] for m in MODEL_ORDER}
    r_ud_test_by_seed: dict[int, np.ndarray] = {}
    for seed in a.seeds:
        print(f"== seed {seed}", flush=True)
        eng = train_userday_engine(tr_ud, cal_ud, seed=seed, epochs=a.epochs, device=a.device,
                                   alpha=FUSION_ALPHA)
        r_cal = risk_scores(eng, cal_ud.x)
        r_te = risk_scores(eng, te_ud.x)
        r_ud_test_by_seed[seed] = r_te["hybrid"]
        x_sup, y_sup = eng.standardise(tr_ud.x), tr_ud.y
        if y_sup.sum() == 0:          # no labelled user-days in the training window
            x_sup, y_sup = eng.standardise(cal_ud.x), cal_ud.y
        for name in SUPERVISED:
            model = fit_baseline(name, x_sup, y_sup, seed=seed)
            r_cal[name] = baseline_risk(model, eng.standardise(cal_ud.x))
            r_te[name] = baseline_risk(model, eng.standardise(te_ud.x))
        for name in MODEL_ORDER:
            # Sensitivity for a user-day: its max request sensitivity; no credit.
            s_cal = cal_ud.x[:, USERDAY_FEATURES.index("max_resource_sensitivity")]
            s_te = te_ud.x[:, USERDAY_FEATURES.index("max_resource_sensitivity")]
            R_cal = effective_risk(r_cal[name], s_cal, 0.0)
            R_te = effective_risk(r_te[name], s_te, 0.0)
            fixed = at_threshold(te_ud.y, R_te, DENY_THRESHOLD)
            tuned = at_threshold(te_ud.y, R_te, best_f1_threshold(cal_ud.y, R_cal))
            per_seed[name].append({"seed": seed, **fixed.as_dict(),
                                   **{f"tuned_{k}": v for k, v in tuned.as_dict().items()}})
            print(f"   {LABELS[name]:20s} user-day AUC {fixed.auc:.3f}  F1@0.85 {fixed.f1:.3f}  "
                  f"tuned F1 {tuned.f1:.3f}", flush=True)
    table_ud = {n: summarise(per_seed[n]) for n in MODEL_ORDER}
    _write_table_v(a.out / "table_v_userday.csv", table_ud)

    # 4. propagate to request level: r' = max(r_request, r_userday[principal, day])
    print("== propagation to request level", flush=True)
    test = load_split(a.data / "test.parquet", max_rows=a.eval_subsample, seed=0,
                      with_types=True, with_keys=True)
    val = load_split(a.data / "val.parquet", with_types=True, with_keys=True)
    k_te = te_ud.key_to_index()
    k_cal = cal_ud.key_to_index()
    idx_te = np.array([k_te.get((p, d), -1) for p, d in zip(test.principal, test.day, strict=True)])
    idx_cal = np.array([k_cal.get((p, d), -1) for p, d in zip(val.principal, val.day, strict=True)])
    per_seed_prop: dict[str, list[dict]] = {"request_only": [], "userday_only": [],
                                            "propagated_max": []}
    for seed in a.seeds:
        req = load_engine(a.models, seed, a.device)
        r_req_te = req.score(test.x, types=test.types).r
        r_req_val = req.score(val.x, types=val.types).r
        eng = train_userday_engine(tr_ud, cal_ud, seed=seed, epochs=a.epochs, device=a.device,
                                   alpha=FUSION_ALPHA)
        r_ud_te = eng.score(te_ud.x).r
        r_ud_cal = eng.score(cal_ud.x).r
        ud_te = np.where(idx_te >= 0, r_ud_te[np.maximum(idx_te, 0)], 0.0)
        ud_cal = np.where(idx_cal >= 0, r_ud_cal[np.maximum(idx_cal, 0)], 0.0)
        variants = {"request_only": (r_req_te, r_req_val), "userday_only": (ud_te, ud_cal),
                    "propagated_max": (np.maximum(r_req_te, ud_te), np.maximum(r_req_val, ud_cal))}
        for name, (r_te_v, r_val_v) in variants.items():
            R_te = effective_risk(r_te_v, test.sensitivity, test.credit)
            R_val = effective_risk(r_val_v, val.sensitivity, val.credit)
            fixed = at_threshold(test.y, R_te, DENY_THRESHOLD)
            tuned = at_threshold(test.y, R_te, best_f1_threshold(val.y, R_val))
            per_seed_prop[name].append({"seed": seed, **fixed.as_dict(),
                                        **{f"tuned_{k}": v for k, v in tuned.as_dict().items()}})
            print(f"   {name:16s} request-level AUC {fixed.auc:.3f}  F1@0.85 {fixed.f1:.3f}  "
                  f"tuned F1 {tuned.f1:.3f}", flush=True)
    table_prop = {n: summarise(v) for n, v in per_seed_prop.items()}
    with (a.out / "table_v_propagated.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "auc", "auc_std", "f1", "fpr_pct", "tuned_f1", "tuned_fpr_pct"])
        for n, t in table_prop.items():
            w.writerow([n, f"{t['auc']:.3f}", f"{t['auc_std']:.3f}", f"{t['f1']:.3f}",
                        f"{100*t['fpr']:.2f}", f"{t['tuned_f1']:.3f}", f"{100*t['tuned_fpr']:.2f}"])
    (a.out / "userday_meta.json").write_text(json.dumps({
        "windows": windows, "seeds": a.seeds, "n_userday_features": len(USERDAY_FEATURES),
        "userdays": {"train": len(tr_ud), "train_pos": int(tr_ud.y.sum()),
                     "calibrate": len(cal_ud), "calibrate_pos": int(cal_ud.y.sum()),
                     "test": len(te_ud), "test_pos": int(te_ud.y.sum())},
        "test_requests": int(len(test)), "test_request_positives": int(test.y.sum()),
        "requests_without_userday": int((idx_te < 0).sum()),
        "per_seed_userday": per_seed, "per_seed_propagated": per_seed_prop,
    }, indent=2))
    print(f"wrote {a.out / 'table_v_userday.csv'} and {a.out / 'table_v_propagated.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
