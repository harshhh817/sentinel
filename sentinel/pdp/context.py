"""Compensating-control credit and feature assembly from a live request.

``control_credit`` is where Module 2's finding 2 is closed: c comes from the
request's authentication context (kappa), not from replayed corpus features. It is
0 unless the device is managed; a managed device earns credit for a fresh MFA
assertion, decaying linearly over eight hours, with a bonus for a hardware-backed
factor. So a stolen key from an unmanaged host gets no relief, and a legitimate user
who just passed hardware MFA on a managed laptop gets up to the full beta = 0.25
discount in eq. (4).

``assemble_features`` (line 2 of Algorithm 1) turns the request into the Module 1
``CloudEvent`` and builds the 34-dim vector against the principal's live baseline.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np

from sentinel.features.baseline import BaselineStore
from sentinel.features.builder import FEATURE_NAMES, build_vector, observe_event
from sentinel.features.schema import CloudEvent
from sentinel.pdp.schemas import AuthContext, AuthorizeRequest

MFA_FRESH_WINDOW_S = 8 * 3600.0
_SENS_IDX = FEATURE_NAMES.index("resource_sensitivity")
_MATCH_IDX = FEATURE_NAMES.index("device_fingerprint_match")
_MFA_IDX = FEATURE_NAMES.index("log_mfa_age")


def control_credit(ctx: AuthContext) -> float:
    """c in [0, 1] from the authentication context."""
    if not ctx.device_managed or ctx.mfa_age_seconds is None:
        return 0.0
    fresh = max(0.0, 1.0 - max(0.0, ctx.mfa_age_seconds) / MFA_FRESH_WINDOW_S)
    base = 0.6 * fresh
    if ctx.mfa_hardware_backed:
        base += 0.4 * fresh
    return float(min(1.0, base))


EGRESS_ACTIONS = {"execute-api:Invoke"}
READ_ACTIONS = {"s3:GetObject", "execute-api:Invoke", "sts:AssumeRole"}


def to_event(req: AuthorizeRequest, now: datetime) -> CloudEvent:
    egress = req.action in EGRESS_ACTIONS or "/removable/" in req.resource
    return CloudEvent(
        event_id="{" + uuid.uuid4().hex[:20].upper() + "}",
        ts=now.replace(tzinfo=None),
        principal=req.principal,
        action=req.action,
        resource=req.resource,
        source="pdp",
        pc=req.context.device_id or req.context.source_ip or "",
        egress=egress,
        bytes_read=int(req.context.bytes_requested or 0),
    )


def assemble_features(
    req: AuthorizeRequest, store: BaselineStore, org: dict[str, str],
    sensitivity: float, now: datetime | None = None,
) -> tuple[np.ndarray, CloudEvent]:
    """Line 2: x <- AssembleFeatures(p, q, rho, kappa, FeatureStore).

    Does NOT update the baseline; that is line 12 and happens only if v != DENY.
    """
    now = now or datetime.now(UTC)
    event = to_event(req, now)
    x = build_vector(event, store.get(req.principal), org)
    # The static policy is the authority on sensitivity; the builder's heuristic is a fallback.
    x[_SENS_IDX] = sensitivity
    # Live context beats the synthesised proxies for the two auth-context features.
    x[_MATCH_IDX] = 1.0 if req.context.device_managed else x[_MATCH_IDX]
    if req.context.mfa_age_seconds is not None:
        x[_MFA_IDX] = float(np.log1p(max(0.0, req.context.mfa_age_seconds)))
    return x, event


def update_baseline(event: CloudEvent, store: BaselineStore) -> None:
    """Line 12: UpdateBaseline(p, x) -- only when v != DENY."""
    observe_event(event, store.get(event.principal))
