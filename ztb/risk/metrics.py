"""Detection metrics for Tables V and VI.

The positive class is under 0.2 % of events, so accuracy is uninformative; we report
precision, recall, F1, ROC-AUC and the false-positive rate at the operating
threshold R >= 0.85 (Table V), plus the same at a threshold tuned for F1 on the
validation window, which is the honest operating point when the paper's fixed one
does not transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ztb.risk.trust import DENY_THRESHOLD


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    auc: float
    fpr: float
    threshold: float
    positives: int
    predicted_positive: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> Metrics:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(score) >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    auc = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return Metrics(precision, recall, f1, auc, fpr, float(threshold), int(y.sum()), int(pred.sum()))


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising F1 on (y, score) -- to be chosen on the validation window."""
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return DENY_THRESHOLD
    fpr, tpr, thr = roc_curve(y, score)
    n_pos, n_neg = y.sum(), (~y).sum()
    tp = tpr * n_pos
    fp = fpr * n_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * tp / (2 * tp + fp + (n_pos - tp))
    f1 = np.nan_to_num(f1)
    best = int(np.argmax(f1))
    t = float(thr[best])
    return min(max(t, 0.0), 1.0) if np.isfinite(t) else DENY_THRESHOLD


def summarise(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean and sample std over seeds for each numeric field."""
    out: dict[str, float] = {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], int | float)]
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0
    return out
