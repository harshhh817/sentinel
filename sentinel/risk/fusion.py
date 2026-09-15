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

from sentinel.config import FUSION_ALPHA
from sentinel.features.builder import FEATURE_NAMES, Standardiser
from sentinel.risk.autoencoder import AutoEncoder, load_autoencoder, save_autoencoder
from sentinel.risk.iforest import isolation_score, load_iforest, save_iforest


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
        cdf_e_by_type: dict[str, EmpiricalCDF] | None = None,
        cdf_s_by_type: dict[str, EmpiricalCDF] | None = None,
    ):
        self.standardiser = standardiser
        self.autoencoder = autoencoder
        self.iforest = iforest
        self.cdf_e = cdf_e
        self.cdf_s = cdf_s
        self.alpha = alpha
        self.feature_names = tuple(feature_names)
        # Per-source calibration: one CDF pair per event type, global as fallback.
        self.cdf_e_by_type = dict(cdf_e_by_type or {})
        self.cdf_s_by_type = dict(cdf_s_by_type or {})

    @property
    def calibration(self) -> str:
        return "per_source" if self.cdf_e_by_type else "global"

    def calibrate_per_type(self, e: np.ndarray, s: np.ndarray, types: np.ndarray,
                           min_rows: int = 200) -> None:
        """Fit one CDF pair per event type from benign calibration scores."""
        self.cdf_e_by_type.clear()
        self.cdf_s_by_type.clear()
        for typ in np.unique(types):
            m = types == typ
            if m.sum() >= min_rows:
                self.cdf_e_by_type[str(typ)] = EmpiricalCDF(e[m])
                self.cdf_s_by_type[str(typ)] = EmpiricalCDF(s[m])

    def _quantiles(self, e: np.ndarray, s: np.ndarray,
                   types: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if types is None or not self.cdf_e_by_type:
            return self.cdf_e(e), self.cdf_s(s)
        f_e, f_s = self.cdf_e(e), self.cdf_s(s)          # global fallback
        for typ, cdf in self.cdf_e_by_type.items():
            m = types == typ
            if m.any():
                f_e[m] = cdf(e[m])
                f_s[m] = self.cdf_s_by_type[typ](s[m])
        return f_e, f_s

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

    def score(self, x_raw: np.ndarray, alpha: float | None = None,
              types: np.ndarray | None = None) -> Scores:
        """``types`` (event type per row) selects the per-source CDFs when fitted."""
        e, s = self.raw_scores(self.standardise(x_raw))
        f_e, f_s = self._quantiles(e, s, types)
        return Scores(e, s, f_e, f_s, fuse(f_e, f_s, self.alpha if alpha is None else alpha))

    # --- persistence -------------------------------------------------------

    def save(self, models_dir: Path, seed: int) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        save_autoencoder(self.autoencoder, models_dir / f"ae_seed{seed}.pt")
        save_iforest(self.iforest, models_dir / f"iforest_seed{seed}.joblib")
        arrays = {"e": self.cdf_e.sorted, "s": self.cdf_s.sorted}
        for typ, cdf in self.cdf_e_by_type.items():
            arrays[f"e__{typ}"] = cdf.sorted
            arrays[f"s__{typ}"] = self.cdf_s_by_type[typ].sorted
        np.savez_compressed(models_dir / f"calib_seed{seed}.npz", **arrays)
        (models_dir / "standardiser.json").write_text(json.dumps(self.standardiser.to_dict()))
        (models_dir / f"engine_seed{seed}.json").write_text(json.dumps({
            "alpha": self.alpha, "feature_names": list(self.feature_names),
            "input_dim": len(self.feature_names), "calibration": self.calibration,
            "types": sorted(self.cdf_e_by_type),
        }))

    @classmethod
    def load(cls, models_dir: Path, seed: int, device: str = "cpu") -> RiskEngine:
        meta = json.loads((models_dir / f"engine_seed{seed}.json").read_text())
        calib = np.load(models_dir / f"calib_seed{seed}.npz")
        by_e = {k[3:]: EmpiricalCDF(calib[k]) for k in calib.files if k.startswith("e__")}
        by_s = {k[3:]: EmpiricalCDF(calib[k]) for k in calib.files if k.startswith("s__")}
        return cls(
            Standardiser.from_dict(json.loads((models_dir / "standardiser.json").read_text())),
            load_autoencoder(models_dir / f"ae_seed{seed}.pt", device),
            load_iforest(models_dir / f"iforest_seed{seed}.joblib"),
            EmpiricalCDF(calib["e"]), EmpiricalCDF(calib["s"]),
            alpha=meta["alpha"], feature_names=tuple(meta["feature_names"]),
            cdf_e_by_type=by_e, cdf_s_by_type=by_s,
        )


class PerSourceEngine:
    """One :class:`RiskEngine` per event type (same architecture and hyperparameters),
    each trained, calibrated and fused on its own kind of event. Types without a model
    fall back to a designated default engine."""

    def __init__(self, engines: dict[str, RiskEngine], default: str):
        if default not in engines:
            raise ValueError(f"default type {default!r} has no engine")
        self.engines = dict(engines)
        self.default = default
        self.alpha = engines[default].alpha
        self.feature_names = engines[default].feature_names

    @property
    def calibration(self) -> str:
        return "per_source_models"

    def standardise(self, x_raw: np.ndarray) -> np.ndarray:
        return self.engines[self.default].standardise(x_raw)

    def score(self, x_raw: np.ndarray, alpha: float | None = None,
              types: np.ndarray | None = None) -> Scores:
        if types is None:
            return self.engines[self.default].score(x_raw, alpha)
        n = len(x_raw)
        e, s, f_e, f_s = (np.zeros(n, np.float32) for _ in range(4))
        for typ in np.unique(types):
            m = types == typ
            eng = self.engines.get(str(typ), self.engines[self.default])
            sc = eng.score(x_raw[m], alpha)
            e[m], s[m], f_e[m], f_s[m] = sc.e, sc.s, sc.f_e, sc.f_s
        return Scores(e, s, f_e, f_s, fuse(f_e, f_s, self.alpha if alpha is None else alpha))

    def save(self, models_dir: Path, seed: int) -> None:
        for typ, eng in self.engines.items():
            eng.save(models_dir / typ, seed)
        (models_dir / f"engine_seed{seed}.json").write_text(json.dumps({
            "alpha": self.alpha, "feature_names": list(self.feature_names),
            "input_dim": len(self.feature_names), "calibration": "per_source_models",
            "types": sorted(self.engines), "default": self.default,
        }))

    @classmethod
    def load(cls, models_dir: Path, seed: int, device: str = "cpu") -> PerSourceEngine:
        meta = json.loads((models_dir / f"engine_seed{seed}.json").read_text())
        engines = {typ: RiskEngine.load(models_dir / typ, seed, device) for typ in meta["types"]}
        return cls(engines, meta["default"])


def load_engine(models_dir: Path, seed: int, device: str = "cpu"):
    """Load whichever engine kind ``models_dir`` holds for this seed."""
    meta = json.loads((models_dir / f"engine_seed{seed}.json").read_text())
    if meta.get("calibration") == "per_source_models":
        return PerSourceEngine.load(models_dir, seed, device)
    return RiskEngine.load(models_dir, seed, device)


def seeds_available(models_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("seed")[1]) for p in models_dir.glob("engine_seed*.json"))
