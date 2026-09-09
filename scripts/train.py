#!/usr/bin/env python
"""Train the hybrid risk engine over five seeds (Section V) -> models/.

For each seed: fit the autoencoder and isolation forest on benign training rows,
early-stop the autoencoder on benign validation loss, calibrate both detectors'
empirical CDFs on the benign validation rows, and save the bundle.

    python scripts/train.py --data data/processed --models models
    python scripts/train.py --data /path/to/synthetic --models /tmp/m --epochs 5 --seeds 0

``--subsample`` bounds how many training rows are used (a seeded uniform sample);
the full window is ~19 M rows and the forest needs only 256 per tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import DATA_PROCESSED, FUSION_ALPHA, MODELS, SEEDS  # noqa: E402
from ztb.features.builder import FEATURE_NAMES, Standardiser  # noqa: E402
from ztb.risk.autoencoder import AEConfig, train_autoencoder  # noqa: E402
from ztb.risk.data import Split, load_split, load_standardiser  # noqa: E402
from ztb.risk.fusion import EmpiricalCDF, RiskEngine  # noqa: E402
from ztb.risk.iforest import train_iforest  # noqa: E402

Preprocess = Callable[[Split], np.ndarray]


def subset_standardiser(std: Standardiser, features: tuple[str, ...]) -> Standardiser:
    idx = [FEATURE_NAMES.index(f) for f in features]
    return Standardiser(mean=std.mean[idx], std=std.std[idx], mask=std.mask[idx])


def train_engine(
    data_dir: Path,
    *,
    seed: int,
    epochs: int,
    subsample: int | None,
    val_subsample: int | None,
    alpha: float = FUSION_ALPHA,
    device: str = "auto",
    features: tuple[str, ...] = FEATURE_NAMES,
    preprocess: Preprocess | None = None,
    verbose: bool = False,
) -> tuple[RiskEngine, dict]:
    """Fit one seed. ``features``/``preprocess`` exist for the ablations."""
    t0 = time.time()
    need_action = preprocess is not None
    train = load_split(data_dir / "train.parquet", max_rows=subsample, seed=seed,
                       with_action=need_action)
    val = load_split(data_dir / "val.parquet", max_rows=val_subsample, seed=seed,
                     with_action=need_action)
    x_tr = preprocess(train) if preprocess else train.x
    x_va = preprocess(val) if preprocess else val.x
    if features != FEATURE_NAMES:
        idx = [FEATURE_NAMES.index(f) for f in features]
        x_tr, x_va = x_tr[:, idx], x_va[:, idx]

    std = load_standardiser(data_dir)
    if std is None:                      # no standardiser.json next to the data: fit here
        std = Standardiser.fit(train.x.astype(np.float64))
    std = subset_standardiser(std, features)

    x_tr_b = std.transform(x_tr[train.y == 0].astype(np.float64)).astype(np.float32)
    x_va_b = std.transform(x_va[val.y == 0].astype(np.float64)).astype(np.float32)
    if len(x_tr_b) == 0 or len(x_va_b) == 0:
        raise SystemExit("need benign rows in both train and val")

    cfg = AEConfig(widths=(len(features), 24, 16, 8), epochs=epochs)
    ae, log = train_autoencoder(x_tr_b, x_va_b, config=cfg, seed=seed, device=device,
                                verbose=verbose)
    forest = train_iforest(x_tr_b, seed=seed)

    # Calibration on the held-out benign set (validation window, Section IV-C).
    e_cal = ae.reconstruction_error(x_va_b)
    from ztb.risk.iforest import isolation_score
    s_cal = isolation_score(forest, x_va_b)
    engine = RiskEngine(std, ae, forest, EmpiricalCDF(e_cal), EmpiricalCDF(s_cal),
                        alpha=alpha, feature_names=features)
    info = {
        "seed": seed, "train_rows_benign": int(len(x_tr_b)), "val_rows_benign": int(len(x_va_b)),
        "epochs_run": log.epochs_run, "best_epoch": log.best_epoch,
        "best_val_loss": log.best_val_loss, "input_dim": len(features),
        "seconds": round(time.time() - t0, 1),
    }
    return engine, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=AEConfig().epochs)
    ap.add_argument("--subsample", type=int, default=1_000_000)
    ap.add_argument("--val-subsample", type=int, default=500_000)
    ap.add_argument("--alpha", type=float, default=FUSION_ALPHA)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not (a.data / "train.parquet").exists():
        ap.error(f"{a.data}/train.parquet not found (run scripts/build_dataset.py first)")

    a.models.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in a.seeds:
        print(f"== seed {seed}", flush=True)
        engine, info = train_engine(a.data, seed=seed, epochs=a.epochs, subsample=a.subsample,
                                    val_subsample=a.val_subsample, alpha=a.alpha,
                                    device=a.device, verbose=a.verbose)
        engine.save(a.models, seed)
        print(f"   {info}", flush=True)
        runs.append(info)
    (a.models / "manifest.json").write_text(json.dumps({
        "data": str(a.data), "alpha": a.alpha, "seeds": a.seeds, "epochs": a.epochs,
        "subsample": a.subsample, "runs": runs,
    }, indent=2))
    print(f"saved {len(runs)} engines -> {a.models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
