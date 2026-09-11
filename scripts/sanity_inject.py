#!/usr/bin/env python
"""Pipeline sanity check: inject synthetic anomalies into the validation window and
confirm the trained autoencoder and forest catch them -> --out/sanity_injection.csv.

If gross anomalies (3 AM removable-media events at 50x the principal's usual volume)
do not separate from benign rows, the scoring pipeline is broken; if they do, the
detectors work and the difficulty is in the data, not the code.

    python scripts/sanity_inject.py --data <splits> --models models --out results
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import DATA_PROCESSED, MODELS, RESULTS  # noqa: E402
from ztb.features.builder import FEATURE_NAMES  # noqa: E402
from ztb.risk.data import load_split  # noqa: E402
from ztb.risk.fusion import load_engine  # noqa: E402

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _set_hour(x: np.ndarray, hour: float) -> None:
    ang = 2 * math.pi * hour / 24.0
    x[:, IDX["hour_sin"]] = math.sin(ang)
    x[:, IDX["hour_cos"]] = math.cos(ang)
    x[:, IDX["modal_hour_deviation"]] = 0.75      # ~9 h from a daytime modal hour


def _scale_volume(x: np.ndarray, k: float) -> None:
    for f in ("calls_1min", "calls_15min", "calls_60min", "distinct_resources_60min"):
        x[:, IDX[f]] = np.maximum(1.0, x[:, IDX[f]]) * k
    x[:, IDX["log_bytes_read_60min"]] += math.log(k)


def injections(base: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    a = base.copy()
    _set_hour(a, 3.0)
    _scale_volume(a, 50.0)
    a[:, IDX["is_egress_action"]] = 1.0
    a[:, IDX["resource_sensitivity"]] = 1.0
    out["3am_egress_50x_volume"] = a
    b = base.copy()
    _scale_volume(b, 50.0)
    out["50x_volume_only"] = b
    c = base.copy()
    _set_hour(c, 3.0)
    out["3am_only"] = c
    d = base.copy()
    for f in ("first_time_action", "first_time_resource", "asn_novelty", "host_novelty"):
        d[:, IDX[f]] = 1.0
    d[:, IDX["device_fingerprint_match"]] = 0.0
    d[:, IDX["geodesic_distance_km"]] = 8000.0
    out["novel_host_and_action"] = d
    e = base + rng.normal(scale=3.0, size=base.shape).astype(np.float32)
    out["gaussian_noise_3sigma"] = e
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20_000, help="benign rows to perturb")
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    val = load_split(a.data / "val.parquet", max_rows=400_000, seed=1, with_types=True).benign
    pick = rng.choice(len(val), size=min(a.n, len(val)), replace=False)
    base, base_types = val.x[pick], val.types[pick]
    rest = np.setdiff1d(np.arange(len(val)), pick)[:200_000]
    engine = load_engine(a.models, a.seed)
    ref = engine.score(val.x[rest], types=val.types[rest])

    rows = []
    for name, xi in injections(base, rng).items():
        sc = engine.score(xi, types=base_types)
        for det, r_ref, r_inj in (("autoencoder", ref.f_e, sc.f_e), ("iforest", ref.f_s, sc.f_s),
                                  ("hybrid", ref.r, sc.r)):
            y = np.r_[np.zeros(len(r_ref)), np.ones(len(r_inj))]
            s = np.r_[r_ref, r_inj]
            rows.append({"injection": name, "detector": det,
                         "auc": round(float(roc_auc_score(y, s)), 4),
                         "injected_r_mean": round(float(r_inj.mean()), 4),
                         "benign_r_mean": round(float(r_ref.mean()), 4),
                         "frac_injected_r_ge_0.85": round(float((r_inj >= 0.85).mean()), 4)})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "sanity_injection.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'injection':26s} {'detector':12s} {'AUC':>6s} {'r_inj':>6s} {'r_ben':>6s} "
          f"{'>=.85':>6s}")
    for r in rows:
        print(f"{r['injection']:26s} {r['detector']:12s} {r['auc']:6.3f} "
              f"{r['injected_r_mean']:6.3f} {r['benign_r_mean']:6.3f} "
              f"{r['frac_injected_r_ge_0.85']:6.3f}")
    print(f"wrote {a.out / 'sanity_injection.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
