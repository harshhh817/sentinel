"""Request/response schemas for POST /authorize."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthContext(BaseModel):
    """kappa: what the PEP learned during primary authentication and posture checks."""

    source_ip: str | None = None
    asn: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    device_id: str = Field(default="", description="device fingerprint / host id")
    device_managed: bool = False
    mfa_age_seconds: float | None = Field(default=None, description="None = no MFA assertion")
    mfa_hardware_backed: bool = False
    auth_class: str = Field(default="password", description="e.g. password, mfa, hw-mfa")
    bytes_requested: int = 0
    step_up_token: str | None = Field(default=None, description="signed challenge from a STEPUP")


class AuthorizeRequest(BaseModel):
    principal: str
    action: str
    resource: str
    context: AuthContext = Field(default_factory=AuthContext)


class RiskDetail(BaseModel):
    e: float | None
    s: float | None
    r: float | None
    R: float | None
    sensitivity: float
    credit: float


class Credential(BaseModel):
    token: str
    scope: str
    ttl_minutes: int
    expires_at: str
    kind: str  # "mock" | "sts"


class AuthorizeResponse(BaseModel):
    verdict: str
    reason: str
    ttl_minutes: int | None
    scope: str | None
    credential: Credential | None
    challenge: str | None = None
    record: dict[str, Any]
    risk: RiskDetail
    timings_ms: dict[str, float]
