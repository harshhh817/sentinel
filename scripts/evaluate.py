#!/usr/bin/env python
"""Reproduce Table V and Figs. 3-4 on the held-out test window -> --out.

Models (identical features and splits): the proposed hybrid, autoencoder-only
(alpha = 1), isolation-forest-only (alpha = 0), and the two supervised baselines,
logistic regression and random forest, trained with malicious labels from the
training window's scripted scenarios (train_malicious.parquet).

Every model yields r in [0, 1]; all pass through the same trust transform (eq. 4) and
are reported at the paper's operating threshold R >= 0.85, and additionally at a
threshold tuned for F1 on the validation window.

    python scripts/evaluate.py --data data/processed --models models --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.config import DATA_PROCESSED, MODELS, RESULTS  # noqa: E402
from sentinel.risk.baselines import SUPERVISED, baseline_risk, fit_baseline  # noqa: E402
from sentinel.risk.data import load_split  # noqa: E402
from sentinel.risk.fusion import fuse, load_engine, seeds_available  # noqa: E402
from sentinel.risk.metrics import at_threshold, best_f1_threshold, summarise  # noqa: E402
from sentinel.risk.trust import DENY_THRESHOLD, effective_risk  # noqa: E402

MODEL_ORDER = ("logistic_regression", "random_forest", "isolation_forest",
               "autoencoder", "hybrid")
LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "isolation_forest": "Isolation forest",
    "autoencoder": "Deep autoencoder",
    "hybrid": "Proposed hybrid",
}


def risk_scores(engine, x: np.ndarray, types: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """r for the three unsupervised variants from one pass through the detectors."""
    sc = engine.score(x, types=types)
    return {
        "hybrid": sc.r,
        "autoencoder": fuse(sc.f_e, sc.f_s, 1.0),
        "isolation_forest": fuse(sc.f_e, sc.f_s, 0.0),
    }


def supervised_training_set(data: Path, seed: int, subsample: int):
    """Benign train subsample + every train-window malicious row (+ val if none)."""
    benign = load_split(data / "train.parquet", max_rows=subsample, seed=seed)
    tm_path = data / "train_malicious.parquet"
    mal = load_split(tm_path) if tm_path.exists() else None
    if mal is None or len(mal) == 0:
        # Corpus with no scenarios in the training window: fall back to the
        # validation window, the only other labelled pre-test data.
        mal = load_split(data / "val.parquet", seed=seed)
        mal = _positives_only(mal)
        source = "val"
    else:
        source = "train_malicious"
    x = np.vstack([benign.x, mal.x])
    y = np.concatenate([benign.y, mal.y])
    return x, y, source


def _positives_only(split):
    m = split.y == 1
    from sentinel.risk.data import Split
    return Split(split.x[m], split.y[m], split.sensitivity[m], split.credit[m])


def evaluate(
    data: Path,
    models: Path,
    out: Path,
    *,
    seeds: list[int] | None = None,
    device: str = "cpu",
    supervised_subsample: int = 300_000,
    eval_subsample: int | None = None,
    make_figures: bool = True,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds or seeds_available(models)
    if not seeds:
        raise SystemExit(f"no trained engines in {models}; run scripts/train.py")

    val = load_split(data / "val.parquet", with_types=True)
    test = load_split(data / "test.parquet", max_rows=eval_subsample, seed=0, with_types=True)
    print(f"val {len(val):,} rows ({int(val.y.sum())} pos)   "
          f"test {len(test):,} rows ({int(test.y.sum())} pos)", flush=True)

    per_seed: dict[str, list[dict]] = {m: [] for m in MODEL_ORDER}
    roc_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sup_source = None
    for seed in seeds:
        print(f"== seed {seed}", flush=True)
        engine = load_engine(models, seed, device)
        r_val = risk_scores(engine, _select(val.x, engine), val.types)
        r_test = risk_scores(engine, _select(test.x, engine), test.types)

        x_sup, y_sup, sup_source = supervised_training_set(data, seed, supervised_subsample)
        x_sup = engine.standardise(x_sup)
        for name in SUPERVISED:
            model = fit_baseline(name, x_sup, y_sup, seed=seed)
            r_val[name] = baseline_risk(model, engine.standardise(val.x))
            r_test[name] = baseline_risk(model, engine.standardise(test.x))

        for name in MODEL_ORDER:
            R_val = effective_risk(r_val[name], val.sensitivity, val.credit)
            R_test = effective_risk(r_test[name], test.sensitivity, test.credit)
            fixed = at_threshold(test.y, R_test, DENY_THRESHOLD)
            tuned_t = best_f1_threshold(val.y, R_val)
            tuned = at_threshold(test.y, R_test, tuned_t)
            row = {"seed": seed, **fixed.as_dict(),
                   **{f"tuned_{k}": v for k, v in tuned.as_dict().items()}}
            per_seed[name].append(row)
            print(f"   {LABELS[name]:20s} F1@0.85 {fixed.f1:.3f}  AUC {fixed.auc:.3f}  "
                  f"FPR {100*fixed.fpr:.2f}%   | tuned@{tuned_t:.3f} F1 {tuned.f1:.3f}",
                  flush=True)
            if seed == seeds[0] and name != "logistic_regression" and make_figures:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(test.y, R_test)
                roc_store[name] = (fpr, tpr)

    table = {name: summarise(per_seed[name]) for name in MODEL_ORDER}
    _write_table_v(out / "table_v.csv", table)
    (out / "table_v_per_seed.json").write_text(json.dumps(
        {"seeds": seeds, "supervised_labels_from": sup_source, "per_seed": per_seed,
         "test_rows": int(len(test)), "test_positives": int(test.y.sum()),
         "calibration": getattr(engine, "calibration", "global")}, indent=2))
    if make_figures:
        _figures(out, table, roc_store)
    print(f"wrote {out/'table_v.csv'}")
    return table


def _select(x: np.ndarray, engine) -> np.ndarray:
    from sentinel.features.builder import FEATURE_NAMES
    if engine.feature_names == FEATURE_NAMES:
        return x
    return x[:, [FEATURE_NAMES.index(f) for f in engine.feature_names]]


def _write_table_v(path: Path, table: dict[str, dict[str, float]]) -> None:
    cols = ["model", "precision", "recall", "f1", "f1_std", "auc", "auc_std", "fpr_pct",
            "tuned_threshold", "tuned_precision", "tuned_recall", "tuned_f1", "tuned_fpr_pct"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name in MODEL_ORDER:
            t = table[name]
            w.writerow([LABELS[name],
                        f"{t['precision']:.3f}", f"{t['recall']:.3f}", f"{t['f1']:.3f}",
                        f"{t['f1_std']:.3f}", f"{t['auc']:.3f}", f"{t['auc_std']:.3f}",
                        f"{100*t['fpr']:.2f}", f"{t['tuned_threshold']:.3f}",
                        f"{t['tuned_precision']:.3f}", f"{t['tuned_recall']:.3f}",
                        f"{t['tuned_f1']:.3f}", f"{100*t['tuned_fpr']:.2f}"])


def _figures(out: Path, table: dict, roc_store: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [LABELS[m] for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.27
    xs = np.arange(len(names))
    for i, key in enumerate(("precision", "recall", "f1")):
        ax.bar(xs + (i - 1) * w, [table[m][key] for m in MODEL_ORDER], w, label=key.capitalize())
    ax.set_xticks(xs, names, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score at R >= 0.85")
    ax.set_title("Fig. 3  Detection performance on the held-out test window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3.png", dpi=150)
    plt.close(fig)

    if roc_store:
        fig, ax = plt.subplots(figsize=(5, 5))
        for name, (fpr, tpr) in roc_store.items():
            ax.plot(fpr, tpr, label=f"{LABELS[name]} (AUC {table[name]['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("false-positive rate")
        ax.set_ylabel("true-positive rate")
        ax.set_title("Fig. 4  ROC curves")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig4.png", dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--supervised-subsample", type=int, default=300_000)
    ap.add_argument("--eval-subsample", type=int, default=None,
                    help="score only a seeded sample of test rows (default: all)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    evaluate(a.data, a.models, a.out, seeds=a.seeds or None, device=a.device,
             supervised_subsample=a.supervised_subsample, eval_subsample=a.eval_subsample,
             make_figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
