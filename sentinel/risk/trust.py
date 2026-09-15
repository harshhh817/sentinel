"""Trust algorithm (Section IV-D, eq. 4) and the decision tiers of Table III.

    R = 1 - (1 - r)^(1 + lambda * s) * (1 - beta * c)          (as typeset)

with lambda = 1.5, beta = 0.25, s the resource sensitivity and c the
compensating-control credit, both in [0, 1].

.. note::
   Read literally, the typeset formula makes the credit *raise* risk: with r = 0 it
   yields R = beta * c, so a perfectly normal request under strong MFA scores 0.25.
   That contradicts the paper's own stated properties ("a request that is not
   anomalous at all remains at R = 0") and the word "credit". The evident intent is
   that the credit scales the risk down:

       R = [1 - (1 - r)^(1 + lambda * s)] * (1 - beta * c)

   which is monotone non-decreasing in r and s, non-increasing in c, and zero at
   r = 0 for every s and c. That is what :func:`effective_risk` computes by default;
   ``literal=True`` reproduces the typeset expression for comparison.
"""

from __future__ import annotations

import numpy as np

from sentinel.config import BANDS, TRUST_BETA, TRUST_LAMBDA, Band


def effective_risk(
    r: np.ndarray | float,
    s: np.ndarray | float,
    c: np.ndarray | float = 0.0,
    *,
    lam: float = TRUST_LAMBDA,
    beta: float = TRUST_BETA,
    literal: bool = False,
) -> np.ndarray:
    """Effective risk R in [0, 1]. Vectorised over numpy arrays."""
    r = np.clip(np.asarray(r, dtype=np.float64), 0.0, 1.0)
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    survival = np.power(1.0 - r, 1.0 + lam * s)
    if literal:
        out = 1.0 - survival * (1.0 - beta * c)
    else:
        out = (1.0 - survival) * (1.0 - beta * c)
    return np.clip(out, 0.0, 1.0)


def band_lookup(R: float) -> Band:
    """Table III: the band whose lower bound is the largest one <= R."""
    chosen = BANDS[0]
    for band in BANDS:
        if R >= band.lower:
            chosen = band
    return chosen


def verdicts(R: np.ndarray) -> np.ndarray:
    """Vectorised band lookup returning verdict strings."""
    lowers = np.array([b.lower for b in BANDS])
    names = np.array([b.verdict for b in BANDS])
    idx = np.searchsorted(lowers, np.asarray(R, dtype=np.float64), side="right") - 1
    return names[np.clip(idx, 0, len(BANDS) - 1)]


DENY_THRESHOLD = BANDS[-1].lower   # 0.85: the operating threshold of Table V
