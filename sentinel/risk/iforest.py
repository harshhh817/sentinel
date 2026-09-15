"""Isolation-forest detector (Section IV-C, eq. 2).

200 random trees over subsamples of 256 points. The anomaly score
s(x) = 2^(-E[h(x)] / c(psi)) lies in (0, 1] and is larger for points isolated in fewer
splits. scikit-learn's ``score_samples`` returns the *negative* of that quantity, so
it is negated here to recover the paper's s(x).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel.config import IFOREST_MAX_SAMPLES, IFOREST_N_ESTIMATORS


def train_iforest(
    x_train: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = IFOREST_N_ESTIMATORS,
    max_samples: int = IFOREST_MAX_SAMPLES,
    n_jobs: int = -1,
) -> IsolationForest:
    """Fit on standardised benign rows."""
    forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, len(x_train)),
        contamination="auto",
        random_state=seed,
        n_jobs=n_jobs,
    )
    forest.fit(x_train)
    return forest


def isolation_score(
    forest: IsolationForest, x: np.ndarray, batch_size: int = 500_000
) -> np.ndarray:
    """s(x) per eq. (2), in (0, 1]; higher is more anomalous."""
    out = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), batch_size):
        out[i:i + batch_size] = -forest.score_samples(x[i:i + batch_size])
    return out


def save_iforest(forest: IsolationForest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, path, compress=3)


def load_iforest(path: Path) -> IsolationForest:
    return joblib.load(path)
