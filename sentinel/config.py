"""Constants from the paper. Every number here has a citation; do not tune them silently.

Sections and tables refer to Gupta et al. (paper.pdf in the repo root).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# --- mode ------------------------------------------------------------------
# "local" is the default; "cloud" swaps in STS, DynamoDB and S3 Object Lock.
MODE = os.environ.get("SENTINEL_MODE", "local")

# --- feature vector (Table II) --------------------------------------------

FEATURE_GROUPS: dict[str, int] = {
    "identity": 5,      # principal type, role tenure, policy breadth, cross-account depth
    "temporal": 6,      # hour sin/cos, day-of-week, modal-hour deviation, inter-request gap
    "action": 7,        # per-principal action frequency, read/write ratio, first-time flag, entropy
    "resource": 5,      # sensitivity label, cross-account flag, novelty, ARN prefix depth
    "volume": 6,        # calls in 1/15/60-min sliding windows, bytes read/60min, distinct resources
    "network": 5,       # ASN novelty, geodesic distance, device-fingerprint match, MFA age
}
N_FEATURES = sum(FEATURE_GROUPS.values())  # == 34

# Baseline update: EWMA with a 30-day effective half-life (Section IV-B).
BASELINE_HALF_LIFE_DAYS = 30.0

# --- risk engine (Section IV-C) -------------------------------------------

AE_ENCODER_WIDTHS = (34, 24, 16, 8)   # decoder is mirrored
AE_DROPOUT = 0.2
AE_EPOCHS = 120
AE_BATCH_SIZE = 512
AE_LEARNING_RATE = 1e-3               # Adam
AE_EARLY_STOPPING_PATIENCE = 10

IFOREST_N_ESTIMATORS = 200
IFOREST_MAX_SAMPLES = 256

FUSION_ALPHA = 0.6                    # eq. (3); grid-searched on the validation split
SEEDS = (0, 1, 2, 3, 4)               # all figures are the mean of five seeds

# --- trust algorithm (eq. 4) ----------------------------------------------

TRUST_LAMBDA = 1.5                    # sensitivity exponent
TRUST_BETA = 0.25                     # compensating-control credit weight


@dataclass(frozen=True)
class Band:
    """One row of Table III."""

    lower: float          # inclusive
    verdict: str
    ttl_minutes: int | None
    scope: str


# Table III, ordered by ascending effective risk R.
BANDS: tuple[Band, ...] = (
    Band(0.00, "ALLOW", 60, "as_requested"),
    Band(0.40, "ALLOW_OBSERVE", 30, "as_requested_verbose"),
    Band(0.65, "STEPUP", 15, "read_only"),
    Band(0.85, "DENY", None, "none"),
)

# --- data split (Section V) -----------------------------------------------
# Split BY TIME, never at random: 10 months train (benign only), 2 validation, 5 test.
SPLIT_MONTHS = {"train": 10, "val": 2, "test": 5}

# --- audit record (eq. 5) --------------------------------------------------

SIGNATURE_CURVE = "P-256"             # ECDSA over NIST P-256
HASH_ALGORITHM = "sha256"
