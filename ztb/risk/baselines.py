"""Supervised baselines of Table V: logistic regression and random forest.

Both are trained *with* malicious labels, which the paper calls "an optimistic upper
bound for supervised methods on this data". Their predicted probability plays the
role of r, so they pass through the same trust transform and operating threshold as
the unsupervised detectors and are evaluated on identical features and splits.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SUPERVISED = ("logistic_regression", "random_forest")


def make_baseline(name: str, seed: int = 0, n_jobs: int = -1):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2,
            random_state=seed, n_jobs=n_jobs,
        )
    raise ValueError(f"unknown baseline {name!r}")


def fit_baseline(name: str, x: np.ndarray, y: np.ndarray, *, seed: int = 0):
    model = make_baseline(name, seed)
    model.fit(x, y)
    return model


def baseline_risk(model, x: np.ndarray, batch_size: int = 500_000) -> np.ndarray:
    """P(malicious) per row, used as r."""
    out = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), batch_size):
        out[i:i + batch_size] = model.predict_proba(x[i:i + batch_size])[:, 1]
    return out
