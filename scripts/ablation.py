#!/usr/bin/env python
"""Reproduce Table VI (ablation and sensitivity) -> --out/table_vi.csv.

Eight rows: the full system and seven variants.

  re-fused from the trained engines (no retraining):
    without isolation forest (alpha = 1.0), without autoencoder (alpha = 0.0),
    alpha = 0.4, alpha = 0.8
  re-scored through the trust transform:
    without resource sensitivity in eq. (4)  (s = 0)
  retrained per seed:
    global instead of per-principal action frequency
    without sliding-window rate features (the six volume-and-rate dims are dropped)

    python scripts/ablation.py --data data/processed --models models --out results
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

from train import train_engine  # noqa: E402

from ztb.config import DATA_PROCESSED, FUSION_ALPHA, MODELS, RESULTS  # noqa: E402
from ztb.features.builder import FEATURE_NAMES, VOLUME_FEATURES  # noqa: E402
from ztb.risk.autoencoder import AEConfig  # noqa: E402
from ztb.risk.data import Split, feature_index, load_split  # noqa: E402
from ztb.risk.fusion import PerSourceEngine, fuse, load_engine, seeds_available  # noqa: E402
from ztb.risk.metrics import at_threshold, best_f1_threshold, summarise  # noqa: E402
from ztb.risk.trust import DENY_THRESHOLD, effective_risk  # noqa: E402

VARIANTS = (
    ("full", "Full system (alpha = 0.6)"),
    ("no_iforest", "Without isolation forest (alpha = 1.0)"),
    ("no_ae", "Without autoencoder (alpha = 0.0)"),
    ("global_action_freq", "Global instead of per-principal action frequency"),
    ("no_rate_features", "Without sliding-window rate features"),
    ("no_sensitivity", "Without resource sensitivity in eq. (4)"),
    ("alpha_0.4", "Fusion weight alpha = 0.4"),
    ("alpha_0.8", "Fusion weight alpha = 0.8"),
)
NO_RATE = tuple(f for f in FEATURE_NAMES if f not in VOLUME_FEATURES)


def global_action_frequency(train_actions: np.ndarray):
    """Replace the per-principal action frequency with the population-wide one."""
    vals, counts = np.unique(train_actions, return_counts=True)
    freq = dict(zip(vals.tolist(), (counts / counts.sum()).tolist(), strict=True))
    col = feature_index("action_frequency")

    def preprocess(split: Split) -> np.ndarray:
        x = split.x.copy()
        if split.action is None:
            raise ValueError("global_action_frequency needs the action column")
        x[:, col] = np.array([freq.get(a, 0.0) for a in split.action], dtype=np.float32)
        return x

    return preprocess


def _metrics(y, r, sens, cred, y_val, r_val, sens_val, cred_val, *, s_off=False):
    s, s_val = (np.zeros_like(sens), np.zeros_like(sens_val)) if s_off else (sens, sens_val)
    R = effective_risk(r, s, cred)
    R_val = effective_risk(r_val, s_val, cred_val)
    fixed = at_threshold(y, R, DENY_THRESHOLD)
    tuned = at_threshold(y, R, best_f1_threshold(y_val, R_val))
    return {"f1": fixed.f1, "auc": fixed.auc, "fpr": fixed.fpr, "tuned_f1": tuned.f1}


def ablation(
    data: Path, models: Path, out: Path, *, seeds: list[int] | None = None,
    epochs: int = AEConfig().epochs, subsample: int = 1_000_000,
    val_subsample: int = 500_000, device: str = "cpu", eval_subsample: int | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or seeds_available(models)
    if not seeds:
        raise SystemExit(f"no trained engines in {models}; run scripts/train.py")

    val = load_split(data / "val.parquet", with_action=True, with_types=True)
    test = load_split(data / "test.parquet", max_rows=eval_subsample, seed=0, with_action=True,
                      with_types=True)
    train_actions = load_split(data / "train.parquet", max_rows=subsample, seed=0,
                               with_action=True).action
    results: dict[str, list[dict]] = {k: [] for k, _ in VARIANTS}

    for seed in seeds:
        print(f"== seed {seed}", flush=True)
        engine = load_engine(models, seed, device)
        if isinstance(engine, PerSourceEngine):
            raise SystemExit("ablation is defined for a single engine; per-source models "
                             "report Table V only (see scripts/evaluate.py)")
        calibration = engine.calibration
        sv, st = engine.score(val.x, types=val.types), engine.score(test.x, types=test.types)
        m = lambda r_t, r_v, **kw: _metrics(  # noqa: E731
            test.y, r_t, test.sensitivity, test.credit,
            val.y, r_v, val.sensitivity, val.credit, **kw)

        for key, alpha in (("full", engine.alpha), ("no_iforest", 1.0), ("no_ae", 0.0),
                           ("alpha_0.4", 0.4), ("alpha_0.8", 0.8)):
            results[key].append(m(fuse(st.f_e, st.f_s, alpha), fuse(sv.f_e, sv.f_s, alpha)))
        results["no_sensitivity"].append(m(st.r, sv.r, s_off=True))

        # retrained variants
        pre = global_action_frequency(train_actions)
        eng_g, _ = train_engine(data, seed=seed, epochs=epochs, subsample=subsample,
                                val_subsample=val_subsample, alpha=FUSION_ALPHA,
                                device=device, preprocess=pre, calibration=calibration)
        results["global_action_freq"].append(
            m(eng_g.score(pre(test), types=test.types).r, eng_g.score(pre(val), types=val.types).r))

        eng_r, _ = train_engine(data, seed=seed, epochs=epochs, subsample=subsample,
                                val_subsample=val_subsample, alpha=FUSION_ALPHA,
                                device=device, features=NO_RATE, calibration=calibration)
        idx = [FEATURE_NAMES.index(f) for f in NO_RATE]
        results["no_rate_features"].append(
            m(eng_r.score(test.x[:, idx], types=test.types).r,
              eng_r.score(val.x[:, idx], types=val.types).r))

        for key, _ in VARIANTS:
            print(f"   {key:22s} F1 {results[key][-1]['f1']:.3f}  "
                  f"AUC {results[key][-1]['auc']:.3f}", flush=True)

    table = {k: summarise(v) for k, v in results.items()}
    base = table["full"]["f1"]
    with (out / "table_vi.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "f1", "f1_std", "delta_f1", "auc", "fpr_pct", "tuned_f1"])
        for key, label in VARIANTS:
            t = table[key]
            w.writerow([label, f"{t['f1']:.3f}", f"{t['f1_std']:.3f}",
                        "baseline" if key == "full" else f"{t['f1'] - base:+.3f}",
                        f"{t['auc']:.3f}", f"{100*t['fpr']:.2f}", f"{t['tuned_f1']:.3f}"])
    (out / "table_vi_per_seed.json").write_text(json.dumps({"seeds": seeds, "per_seed": results},
                                                           indent=2))
    print(f"wrote {out/'table_vi.csv'}")
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=AEConfig().epochs)
    ap.add_argument("--subsample", type=int, default=1_000_000)
    ap.add_argument("--val-subsample", type=int, default=500_000)
    ap.add_argument("--eval-subsample", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    ablation(a.data, a.models, a.out, seeds=a.seeds or None, epochs=a.epochs,
             subsample=a.subsample, val_subsample=a.val_subsample, device=a.device,
             eval_subsample=a.eval_subsample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
