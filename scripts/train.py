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

from sentinel.config import DATA_PROCESSED, FUSION_ALPHA, MODELS, SEEDS  # noqa: E402
from sentinel.features.builder import FEATURE_NAMES, Standardiser  # noqa: E402
from sentinel.risk.autoencoder import AEConfig, train_autoencoder  # noqa: E402
from sentinel.risk.data import Split, load_split, load_standardiser  # noqa: E402
from sentinel.risk.fusion import EmpiricalCDF, PerSourceEngine, RiskEngine  # noqa: E402
from sentinel.risk.iforest import isolation_score, train_iforest  # noqa: E402
from sentinel.risk.types import EVENT_TYPES  # noqa: E402

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
    calibration: str = "global",
    only_type: str | None = None,
    verbose: bool = False,
) -> tuple[RiskEngine, dict]:
    """Fit one seed. ``features``/``preprocess`` exist for the ablations.

    ``calibration="per_source"`` additionally fits one CDF pair per event type on the
    benign validation rows of that type. ``only_type`` restricts training *and*
    calibration to one event type (used to build a :class:`PerSourceEngine`).
    """
    t0 = time.time()
    need_types = calibration == "per_source" or only_type is not None
    train = load_split(data_dir / "train.parquet", max_rows=subsample, seed=seed,
                       with_action=preprocess is not None, with_types=need_types)
    val = load_split(data_dir / "val.parquet", max_rows=val_subsample, seed=seed,
                     with_action=preprocess is not None, with_types=need_types)
    if only_type is not None:
        train, val = train.subset(train.types == only_type), val.subset(val.types == only_type)
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
    s_cal = isolation_score(forest, x_va_b)
    engine = RiskEngine(std, ae, forest, EmpiricalCDF(e_cal), EmpiricalCDF(s_cal),
                        alpha=alpha, feature_names=features)
    if calibration == "per_source":
        engine.calibrate_per_type(e_cal, s_cal, val.types[val.y == 0])
    info = {
        "seed": seed, "train_rows_benign": int(len(x_tr_b)), "val_rows_benign": int(len(x_va_b)),
        "epochs_run": log.epochs_run, "best_epoch": log.best_epoch,
        "best_val_loss": log.best_val_loss, "input_dim": len(features),
        "calibration": engine.calibration, "type": only_type,
        "seconds": round(time.time() - t0, 1),
    }
    return engine, info


def recalibrate(models_dir: Path, data_dir: Path, seed: int, *, val_subsample: int | None,
                per_source: bool = True) -> tuple[RiskEngine, dict]:
    """Refit the calibration CDFs of an existing engine without retraining."""
    engine = RiskEngine.load(models_dir, seed)
    val = load_split(data_dir / "val.parquet", max_rows=val_subsample, seed=seed, with_types=True)
    b = val.benign
    e, s = engine.raw_scores(engine.standardise(b.x))
    engine.cdf_e, engine.cdf_s = EmpiricalCDF(e), EmpiricalCDF(s)
    if per_source:
        engine.calibrate_per_type(e, s, b.types)
    else:
        engine.cdf_e_by_type.clear()
        engine.cdf_s_by_type.clear()
    return engine, {"seed": seed, "val_rows_benign": int(len(b)), "calibration": engine.calibration,
                    "types": sorted(engine.cdf_e_by_type)}


def train_per_source(data_dir: Path, *, seed: int, epochs: int, subsample: int | None,
                     val_subsample: int | None, alpha: float, device: str,
                     min_rows: int = 2000, verbose: bool = False) -> tuple[PerSourceEngine, dict]:
    """One engine per event type present in the data (same architecture/hyperparameters)."""
    probe = load_split(data_dir / "train.parquet", max_rows=200_000, seed=seed, with_types=True)
    counts = {str(t): int((probe.types == t).sum()) for t in np.unique(probe.types)}
    engines, infos = {}, []
    for typ in EVENT_TYPES:
        if counts.get(typ, 0) * (1 if not subsample else 1) < min_rows * 0.2:
            continue                      # too rare in this corpus to fit its own manifold
        eng, info = train_engine(data_dir, seed=seed, epochs=epochs, subsample=subsample,
                                 val_subsample=val_subsample, alpha=alpha, device=device,
                                 only_type=typ, verbose=verbose)
        engines[typ] = eng
        infos.append(info)
        print(f"   {typ:9s} {info}", flush=True)
    default = max(engines, key=lambda k: counts.get(k, 0))
    return PerSourceEngine(engines, default), {"seed": seed, "types": sorted(engines),
                                              "default": default, "runs": infos}


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
    ap.add_argument("--calibration", choices=["global", "per_source"], default="global",
                    help="per_source: one CDF pair per event type (global as fallback)")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refit calibration of the engines already in --models; no retraining")
    ap.add_argument("--per-source-models", action="store_true",
                    help="train one autoencoder + forest per event type")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if not (a.data / "train.parquet").exists():
        ap.error(f"{a.data}/train.parquet not found (run scripts/build_dataset.py first)")

    a.models.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in a.seeds:
        print(f"== seed {seed}", flush=True)
        if a.recalibrate:
            engine, info = recalibrate(a.models, a.data, seed, val_subsample=a.val_subsample,
                                       per_source=a.calibration == "per_source")
        elif a.per_source_models:
            engine, info = train_per_source(a.data, seed=seed, epochs=a.epochs,
                                            subsample=a.subsample, val_subsample=a.val_subsample,
                                            alpha=a.alpha, device=a.device, verbose=a.verbose)
        else:
            engine, info = train_engine(a.data, seed=seed, epochs=a.epochs, subsample=a.subsample,
                                        val_subsample=a.val_subsample, alpha=a.alpha,
                                        device=a.device, calibration=a.calibration,
                                        verbose=a.verbose)
        engine.save(a.models, seed)
        print(f"   {info}", flush=True)
        runs.append(info)
    (a.models / "manifest.json").write_text(json.dumps({
        "data": str(a.data), "alpha": a.alpha, "seeds": a.seeds, "epochs": a.epochs,
        "subsample": a.subsample, "calibration": a.calibration,
        "per_source_models": a.per_source_models, "runs": runs,
    }, indent=2, default=str))
    print(f"saved {len(runs)} engines -> {a.models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
