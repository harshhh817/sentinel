"""Build the 34-dimensional behavioural feature vector (Table II).

The vector is assembled from three sources: the request itself, the principal's
:class:`~sentinel.features.baseline.PrincipalProfile` (their decayed history and sliding
windows) and the static org data from LDAP. Group sizes are fixed by Table II and
asserted at import: 5 identity, 6 temporal, 7 action, 5 resource, 6 volume, 5 network.

Encoding rules from Section IV-B:

* cyclic quantities (hour, day-of-week) become sine/cosine pairs, so 23:00 and 01:00
  are close;
* high-cardinality categoricals (the API action, the resource) are replaced by their
  empirical frequency **in the principal's own history**, which bounds dimensionality
  and directly expresses how unusual the call is for that user;
* continuous features are standardised, with statistics fitted on the training window
  only — see :class:`Standardiser`.

.. warning::
   CERT r4.2 carries no network telemetry: there is no source address, ASN,
   geolocation, device fingerprint or MFA assertion in the corpus. The five
   network-and-device features are therefore **synthesised deterministically from the
   originating host** (:func:`host_identity`), so that switching PC moves a principal
   to a different pseudo-ASN and pseudo-location while a stable PC stays put. This
   preserves the novelty/continuity signal those features are meant to carry, but it is
   an augmentation the corpus does not support, and it is a further approximation on
   top of the CERT-to-CloudTrail mapping the paper already flags as its main threat to
   external validity.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from sentinel.config import FEATURE_GROUPS, N_FEATURES
from sentinel.features.baseline import PrincipalProfile
from sentinel.features.schema import CloudEvent

# --- feature names, in vector order ---------------------------------------

IDENTITY_FEATURES = (
    "principal_type",         # 0 human user, 1 service/admin role
    "role_tenure_days",       # days since the principal was first observed
    "policy_breadth",         # distinct actions the principal has ever been permitted
    "cross_account_depth",    # org-tree depth of the principal's unit
    "is_supervisor",          # supervises others -> broader legitimate reach
)

TEMPORAL_FEATURES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "modal_hour_deviation",   # circular distance from the principal's modal hour
    "log_inter_request_gap",  # log1p seconds since the principal's previous request
)

ACTION_FEATURES = (
    "action_frequency",       # per-principal empirical frequency of this action
    "read_write_ratio",
    "first_time_action",
    "action_class_entropy",
    "log_action_history",     # log1p of the principal's decayed total call count
    "is_session_action",      # sts:AssumeRole / sts:SessionEnd
    "is_egress_action",       # removable media or external proxy
)

RESOURCE_FEATURES = (
    "resource_sensitivity",   # s in eq. (4), in [0, 1]
    "cross_account_flag",     # resource outside the principal's own unit bucket
    "resource_novelty",       # 1 - empirical frequency for this principal
    "arn_prefix_depth",
    "first_time_resource",
)

VOLUME_FEATURES = (
    "calls_1min",
    "calls_15min",
    "calls_60min",
    "log_bytes_read_60min",
    "distinct_resources_60min",
    "distinct_actions_60min",
)

NETWORK_FEATURES = (
    "asn_novelty",            # 1 if this pseudo-ASN is new to the principal
    "geodesic_distance_km",   # great-circle distance from the previous request
    "device_fingerprint_match",
    "log_mfa_age",            # log1p seconds since last logon
    "host_novelty",
)

FEATURE_NAMES: tuple[str, ...] = (
    IDENTITY_FEATURES + TEMPORAL_FEATURES + ACTION_FEATURES
    + RESOURCE_FEATURES + VOLUME_FEATURES + NETWORK_FEATURES
)

# Table II is the specification; a drift in either direction is a bug.
assert len(IDENTITY_FEATURES) == FEATURE_GROUPS["identity"]
assert len(TEMPORAL_FEATURES) == FEATURE_GROUPS["temporal"]
assert len(ACTION_FEATURES) == FEATURE_GROUPS["action"]
assert len(RESOURCE_FEATURES) == FEATURE_GROUPS["resource"]
assert len(VOLUME_FEATURES) == FEATURE_GROUPS["volume"]
assert len(NETWORK_FEATURES) == FEATURE_GROUPS["network"]
assert len(FEATURE_NAMES) == N_FEATURES == 34
assert len(set(FEATURE_NAMES)) == N_FEATURES, "duplicate feature name"

# Continuous features standardised over the training window. Flags, frequencies and
# sin/cos pairs are already bounded and are left alone.
CONTINUOUS_FEATURES: frozenset[str] = frozenset({
    "role_tenure_days", "policy_breadth", "cross_account_depth",
    "log_inter_request_gap", "log_action_history", "arn_prefix_depth",
    "calls_1min", "calls_15min", "calls_60min", "log_bytes_read_60min",
    "distinct_resources_60min", "distinct_actions_60min",
    "geodesic_distance_km", "log_mfa_age",
})

# --- resource sensitivity --------------------------------------------------
# s in eq. (4). The corpus has no sensitivity labels, so they are assigned by resource
# kind: anything leaving the organisation ranks highest.

SENSITIVITY: dict[str, float] = {
    "removable": 1.0,     # egress to removable media
    "external": 0.8,      # external-service proxy
    "session": 0.3,       # role assumption / session end
    "bucket": 0.6,        # named OU bucket object
    "default": 0.5,
}

READ_ACTIONS = frozenset({"s3:GetObject", "execute-api:Invoke", "sts:AssumeRole"})
SESSION_ACTIONS = frozenset({"sts:AssumeRole", "sts:SessionEnd"})

EARTH_RADIUS_KM = 6371.0


# --- synthetic network identity -------------------------------------------


def host_identity(pc: str) -> tuple[str, float, float]:
    """Deterministic ``(asn, latitude, longitude)`` for a CERT host.

    A hash keeps this stable across runs and machines, so a principal who always uses
    the same PC shows zero movement and one who switches shows a real jump.
    """
    if not pc:
        return ("AS0", 0.0, 0.0)
    digest = hashlib.sha256(pc.encode()).digest()
    asn = f"AS{int.from_bytes(digest[:2], 'big')}"
    lat = (int.from_bytes(digest[2:5], "big") / 0xFFFFFF) * 180.0 - 90.0
    lon = (int.from_bytes(digest[5:8], "big") / 0xFFFFFF) * 360.0 - 180.0
    return (asn, lat, lon)


def geodesic_km(a: tuple[float, float] | None, b: tuple[float, float]) -> float:
    """Great-circle distance in km; 0 when there is no previous request."""
    if a is None:
        return 0.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def sensitivity_of(event: CloudEvent) -> float:
    """Resource sensitivity s used by the trust algorithm (eq. 4)."""
    if event.egress and "removable" in event.resource:
        return SENSITIVITY["removable"]
    if event.action == "execute-api:Invoke":
        return SENSITIVITY["external"]
    if event.action in SESSION_ACTIONS:
        return SENSITIVITY["session"]
    if event.resource.startswith("arn:aws:s3:::"):
        return SENSITIVITY["bucket"]
    return SENSITIVITY["default"]


# --- standardisation -------------------------------------------------------


@dataclass
class Standardiser:
    """Mean/std for the continuous features, fitted on the training window only.

    Fitting on train and applying to val/test is what keeps the split honest: no
    statistic derived from a user's future ever touches the scoring of their past.
    """

    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray  # True where the feature is standardised

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Standardiser:
        if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
            raise ValueError(f"expected (n, {N_FEATURES}) matrix, got {matrix.shape}")
        mask = np.array([name in CONTINUOUS_FEATURES for name in FEATURE_NAMES])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-8] = 1.0          # constant column -> leave values unchanged
        mean[~mask] = 0.0
        std[~mask] = 1.0
        return cls(mean=mean, std=std, mask=mask)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.std

    def to_dict(self) -> dict[str, list]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "mask": self.mask.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Standardiser:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            mask=np.asarray(payload["mask"], dtype=bool),
        )


class StandardiserAccumulator:
    """Fit a :class:`Standardiser` in one streaming pass, without holding the matrix.

    Uses Welford's online algorithm, so memory is O(34) rather than O(rows). The
    training window is tens of millions of events on the full corpus; materialising it
    to call :meth:`Standardiser.fit` would need gigabytes and defeat the streaming
    design of ``scripts/build_dataset.py``.
    """

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self) -> None:
        self._n = 0
        self._mean = np.zeros(N_FEATURES, dtype=np.float64)
        self._m2 = np.zeros(N_FEATURES, dtype=np.float64)

    def __len__(self) -> int:
        return self._n

    def update(self, vector: np.ndarray) -> None:
        self._n += 1
        delta = vector - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (vector - self._mean)

    def finalize(self) -> Standardiser:
        if self._n == 0:
            raise ValueError("no rows accumulated; cannot fit a standardiser")
        mask = np.array([name in CONTINUOUS_FEATURES for name in FEATURE_NAMES])
        mean = self._mean.copy()
        std = np.sqrt(self._m2 / self._n) if self._n > 1 else np.ones(N_FEATURES)
        std[std < 1e-8] = 1.0
        mean[~mask] = 0.0
        std[~mask] = 1.0
        return Standardiser(mean=mean, std=std, mask=mask)


# --- the builder -----------------------------------------------------------


def build_vector(
    event: CloudEvent,
    profile: PrincipalProfile,
    org: dict[str, str] | None = None,
) -> np.ndarray:
    """Return the raw (unstandardised) 34-dim vector for one event.

    Must be called **before** ``profile.observe(...)`` for this event, so the vector
    describes the request against the principal's history up to but excluding itself.
    """
    org = org or {}
    ts: datetime = event.ts
    asn, lat, lon = host_identity(event.pc)
    is_session = event.action in SESSION_ACTIONS

    # identity and entitlement (5)
    role = (org.get("role") or "").lower()
    principal_type = 1.0 if any(k in role for k in ("admin", "it", "director")) else 0.0
    tenure = profile.tenure_days(ts)
    policy_breadth = float(len(profile.seen_actions))
    unit_depth = float(sum(1 for k in ("business_unit", "functional_unit", "department",
                                       "team") if org.get(k)))
    is_supervisor = 1.0 if (org.get("supervisor") or "").strip() == "" and org else 0.0

    # temporal (6)
    hour_angle = 2 * math.pi * (ts.hour + ts.minute / 60.0) / 24.0
    dow_angle = 2 * math.pi * ts.weekday() / 7.0
    gap = profile.inter_request_gap(ts)
    log_gap = math.log1p(gap) if gap >= 0 else 0.0

    # action (7)
    action_freq = profile.action_frequency(event.action, ts)
    rw_ratio = profile.read_write_ratio(ts)
    first_time_action = 0.0 if event.action in profile.seen_actions else 1.0
    entropy = profile.action_entropy(ts)
    log_history = math.log1p(profile.total)

    # resource (5)
    own_bucket = f"sentinel-{(org.get('functional_unit') or org.get('business_unit') or '')}"
    cross_account = 0.0
    if event.resource.startswith("arn:aws:s3:::") and org:
        bucket = event.resource.split(":::", 1)[1].split("/", 1)[0]
        cross_account = 0.0 if bucket in own_bucket.lower().replace(" ", "-") else 1.0
    resource_novelty = 1.0 - profile.resource_frequency(event.resource, ts)
    prefix_depth = float(event.resource.count("/"))
    first_time_resource = 0.0 if event.resource in profile.seen_resources else 1.0

    # volume and rate (6) — windows anchored on this event's timestamp
    c1, c15, c60, distinct_resources, distinct_actions = profile.window_counts(ts)
    log_bytes = math.log1p(profile.bytes_read_60m(ts))

    # network and device (5) — synthesised, see the module warning
    asn_novelty = 0.0 if asn in profile.seen_asns else 1.0
    distance = geodesic_km(profile.prev_location, (lat, lon))
    device_match = 1.0 if event.pc in profile.seen_pcs else 0.0
    mfa_age = profile.mfa_age_seconds(ts)
    log_mfa_age = math.log1p(mfa_age) if mfa_age >= 0 else 0.0
    host_novelty = 0.0 if event.pc in profile.seen_pcs else 1.0

    vector = np.array([
        # identity
        principal_type, tenure, policy_breadth, unit_depth, is_supervisor,
        # temporal
        math.sin(hour_angle), math.cos(hour_angle),
        math.sin(dow_angle), math.cos(dow_angle),
        profile.modal_hour_deviation(ts), log_gap,
        # action
        action_freq, rw_ratio, first_time_action, entropy, log_history,
        1.0 if is_session else 0.0, 1.0 if event.egress else 0.0,
        # resource
        sensitivity_of(event), cross_account, resource_novelty, prefix_depth,
        first_time_resource,
        # volume
        float(c1), float(c15), float(c60), log_bytes,
        float(distinct_resources), float(distinct_actions),
        # network
        asn_novelty, distance, device_match, log_mfa_age, host_novelty,
    ], dtype=np.float64)

    if vector.shape != (N_FEATURES,):  # pragma: no cover - guarded by tests
        raise AssertionError(f"feature vector is {vector.shape}, expected ({N_FEATURES},)")
    return vector


def observe_event(event: CloudEvent, profile: PrincipalProfile) -> None:
    """Fold an event into its principal's baseline, after its vector has been built."""
    asn, lat, lon = host_identity(event.pc)
    profile.observe(
        event.ts,
        event.action,
        event.resource,
        event.pc,
        asn,
        is_read=event.action in READ_ACTIONS,
        bytes_read=float(event.bytes_read),
        location=(lat, lon),
        is_logon=event.action == "sts:AssumeRole",
    )
