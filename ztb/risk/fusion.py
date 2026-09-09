"""Score fusion (Section IV-C, eq. 3) and the assembled risk engine.

    r(x) = alpha * F_e(e(x)) + (1 - alpha) * F_s(s(x))

where F_e and F_s are empirical CDFs of the autoencoder error and the isolation
score, estimated on a held-out benign calibration set (the benign rows of the
validation window). r is then directly interpretable as the fraction of benign
requests this one exceeds in anomaly.

:class:`RiskEngine` bundles the standardiser, both detectors and both CDFs for one
seed, and can be saved to / loaded from a models directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ztb.config import FUSION_ALPHA
from ztb.features.builder import FEATURE_NAMES, Standardiser
from ztb.risk.autoencoder import AutoEncoder, load_autoencoder, save_autoencoder
from ztb.risk.iforest import isolation_score, load_iforest, save_iforest


class EmpiricalCDF:
    """F(v) = fraction of calibration values <= v. Monotone, in [0, 1]."""

    def __init__(self, values: np.ndarray):
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            raise ValueError("empty calibration set")
        self.sorted = np.sort(v)

    def __call__(self, v: np.ndarray | float) -> np.ndarray:
        idx = np.searchsorted(self.sorted, np.asarray(v, dtype=np.float64), side="right")
        return idx / len(self.sorted)

    def __len__(self) -> int:
        return len(self.sorted)


def fuse(f_e: np.ndarray, f_s: np.ndarray, alpha: float = FUSION_ALPHA) -> np.ndarray:
    """eq. (3). alpha = 1 is autoencoder-only, alpha = 0 is isolation-forest-only."""
    return np.clip(alpha * f_e + (1.0 - alpha) * f_s, 0.0, 1.0)


@dataclass
class Scores:
    e: np.ndarray        # autoencoder reconstruction error
    s: np.ndarray        # isolation score
    f_e: np.ndarray      # calibrated quantile of e
    f_s: np.ndarray      # calibrated quantile of s
    r: np.ndarray        # fused risk


class RiskEngine:
    """Standardise -> score with both detectors -> calibrate -> fuse."""

    def __init__(
        self,
        standardiser: Standardiser,
        autoencoder: AutoEncoder,
        iforest,
        cdf_e: EmpiricalCDF,
        cdf_s: EmpiricalCDF,
        *,
        alpha: float = FUSION_ALPHA,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ):
        self.standardiser = standardiser
        self.autoencoder = autoencoder
        self.iforest = iforest
        self.cdf_e = cdf_e
        self.cdf_s = cdf_s
        self.alpha = alpha
        self.feature_names = tuple(feature_names)

    # --- scoring -----------------------------------------------------------

    def standardise(self, x_raw: np.ndarray) -> np.ndarray:
        # float32 throughout: a float64 round trip doubles the peak on millions of rows.
        x = np.asarray(x_raw, dtype=np.float32)
        mean = self.standardiser.mean.astype(np.float32)
        std = self.standardiser.std.astype(np.float32)
        return (x - mean) / std

    def raw_scores(self, x_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.autoencoder.reconstruction_error(x_std),
                isolation_score(self.iforest, x_std))

    def score(self, x_raw: np.ndarray, alpha: float | None = None) -> Scores:
        e, s = self.raw_scores(self.standardise(x_raw))
        f_e, f_s = self.cdf_e(e), self.cdf_s(s)
        return Scores(e, s, f_e, f_s, fuse(f_e, f_s, self.alpha if alpha is None else alpha))

    # --- persistence -------------------------------------------------------

    def save(self, models_dir: Path, seed: int) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        save_autoencoder(self.autoencoder, models_dir / f"ae_seed{seed}.pt")
        save_iforest(self.iforest, models_dir / f"iforest_seed{seed}.joblib")
        np.savez_compressed(models_dir / f"calib_seed{seed}.npz",
                            e=self.cdf_e.sorted, s=self.cdf_s.sorted)
        (models_dir / "standardiser.json").write_text(json.dumps(self.standardiser.to_dict()))
        (models_dir / f"engine_seed{seed}.json").write_text(json.dumps({
            "alpha": self.alpha, "feature_names": list(self.feature_names),
            "input_dim": len(self.feature_names),
        }))

    @classmethod
    def load(cls, models_dir: Path, seed: int, device: str = "cpu") -> RiskEngine:
        meta = json.loads((models_dir / f"engine_seed{seed}.json").read_text())
        calib = np.load(models_dir / f"calib_seed{seed}.npz")
        return cls(
            Standardiser.from_dict(json.loads((models_dir / "standardiser.json").read_text())),
            load_autoencoder(models_dir / f"ae_seed{seed}.pt", device),
            load_iforest(models_dir / f"iforest_seed{seed}.joblib"),
            EmpiricalCDF(calib["e"]), EmpiricalCDF(calib["s"]),
            alpha=meta["alpha"], feature_names=tuple(meta["feature_names"]),
        )


def seeds_available(models_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("seed")[1]) for p in models_dir.glob("engine_seed*.json"))
