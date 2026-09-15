"""Event types for per-source calibration and per-source models.

CERT replayed as CloudTrail collapses to five kinds of control-plane call. They differ
structurally in the feature space (action, resource, sensitivity, egress flags), so a
single benign calibration dominated by http (94 % of events) places every other kind
near the top of the benign quantile regardless of intent. Calibrating -- or modelling --
per type makes r read as "fraction of benign events *of this kind* this one exceeds".
"""

from __future__ import annotations

import numpy as np

EVENT_TYPES: tuple[str, ...] = ("sts", "s3_read", "s3_write", "egress", "http")
EGRESS_SOURCES = frozenset({"device", "file"})


def event_type(action: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Vectorised type per row from the parquet's ``action`` and ``source`` columns."""
    action = np.asarray(action).astype(str)
    source = np.asarray(source).astype(str)
    out = np.full(len(action), "http", dtype="<U8")
    out[np.char.startswith(action, "sts:")] = "sts"
    put = action == "s3:PutObject"
    get = action == "s3:GetObject"
    egress = np.isin(source, list(EGRESS_SOURCES))
    out[put & ~egress] = "s3_write"
    out[get & ~egress] = "s3_read"
    out[(get | put) & egress] = "egress"
    return out


def event_type_one(action: str, source: str, resource: str = "") -> str:
    """Scalar form for the PDP, where the source is the request itself."""
    if action.startswith("sts:"):
        return "sts"
    if action in ("s3:GetObject", "s3:PutObject"):
        if source in EGRESS_SOURCES or "/removable/" in resource:
            return "egress"
        return "s3_write" if action == "s3:PutObject" else "s3_read"
    return "http"
