"""Scoped, short-lived credentials -- line 7 of Algorithm 1.

Local mode issues a signed mock token carrying principal, scope, TTL and expiry; it
is verifiable with the PDP's public key and is what the local PEP would present to a
mock resource. Cloud mode calls ``sts.assume_role`` with a session policy narrowed
to ``scope`` and ``DurationSeconds = ttl``; that path is behind ``Settings.cloud``
and imports boto3 lazily so local mode has no AWS dependency.
"""

from __future__ import annotations

import base64
import json
import secrets
from datetime import UTC, datetime, timedelta

from sentinel.pdp.schemas import Credential
from sentinel.pdp.signer import Signer, Verifier

SCOPE_POLICIES = {
    # Session-policy fragments keyed by Table III scope.
    "as_requested": {"Effect": "Allow", "Action": ["{action}"], "Resource": ["{resource}"]},
    "as_requested_verbose": {"Effect": "Allow", "Action": ["{action}"], "Resource": ["{resource}"],
                             "Condition": {"Bool": {"ztb:verboseLogging": "true"}}},
    "read_only": {"Effect": "Allow",
                  "Action": ["s3:GetObject", "s3:ListBucket", "sts:GetCallerIdentity"],
                  "Resource": ["{resource}"]},
}


def session_policy(scope: str, action: str, resource: str) -> dict:
    stmt = json.loads(json.dumps(SCOPE_POLICIES[scope]).replace("{action}", action)
                      .replace("{resource}", resource))
    return {"Version": "2012-10-17", "Statement": [stmt]}


def issue_mock(signer: Signer, principal: str, action: str, resource: str,
               scope: str, ttl_minutes: int, now: datetime | None = None) -> Credential:
    now = now or datetime.now(UTC)
    exp = now + timedelta(minutes=ttl_minutes)
    claims = {
        "sub": principal, "scope": scope, "action": action, "resource": resource,
        "iat": now.isoformat(), "exp": exp.isoformat(), "nonce": secrets.token_hex(8),
        "policy": session_policy(scope, action, resource),
    }
    body = base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True,
                                               separators=(",", ":")).encode()).decode()
    sig = signer.sign_bytes(body.encode())
    return Credential(token=f"{body}.{sig}", scope=scope, ttl_minutes=ttl_minutes,
                      expires_at=exp.isoformat(), kind="mock")


def verify_mock(verifier: Verifier, token: str, now: datetime | None = None) -> dict | None:
    """Claims if the token is authentic and unexpired, else None."""
    try:
        body, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    if not verifier.verify_bytes(body.encode(), sig):
        return None
    claims = json.loads(base64.urlsafe_b64decode(body.encode()))
    now = now or datetime.now(UTC)
    if datetime.fromisoformat(claims["exp"]) <= now:
        return None
    return claims


def issue_sts(role_arn: str, principal: str, action: str, resource: str,
              scope: str, ttl_minutes: int) -> Credential:
    """Cloud mode: STS AssumeRole with a session policy. Not exercised locally."""
    import boto3  # lazy: local mode must not need it

    sts = boto3.client("sts")
    resp = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=f"sentinel-{principal}"[:64],
        DurationSeconds=max(900, ttl_minutes * 60),
        Policy=json.dumps(session_policy(scope, action, resource)),
    )
    creds = resp["Credentials"]
    token = json.dumps({"AccessKeyId": creds["AccessKeyId"],
                        "SecretAccessKey": creds["SecretAccessKey"],
                        "SessionToken": creds["SessionToken"]})
    return Credential(token=token, scope=scope, ttl_minutes=ttl_minutes,
                      expires_at=creds["Expiration"].isoformat(), kind="sts")


def issue(settings, signer: Signer, principal: str, action: str, resource: str,
          scope: str, ttl_minutes: int) -> Credential:
    if settings.cloud:
        if not settings.sts_role_arn:
            raise RuntimeError("SENTINEL_STS_ROLE_ARN is required in cloud mode")
        return issue_sts(settings.sts_role_arn, principal, action, resource, scope, ttl_minutes)
    return issue_mock(signer, principal, action, resource, scope, ttl_minutes)
