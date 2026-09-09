"""Policy decision point: POST /authorize implements Algorithm 1 line by line.

    1  if not StaticallyPermitted(p, q, rho) then return DENY
    2  x <- AssembleFeatures(p, q, rho, kappa, FeatureStore)
    3  e <- ||x - g(f(x))||_2 ; s <- IsolationScore(x)
    4  r <- alpha F_e(e) + (1 - alpha) F_s(s)
    5  R <- 1 - (1 - r)^(1 + lambda Sens(rho)) (1 - beta Ctrl(kappa))
    6  <v, ttl, scope> <- BandLookup(R)
    7  sigma <- STS.AssumeRole(p, scope, ttl) if v = ALLOW
    8         ChallengeAndRetry(p, q)        if v = STEPUP, else bottom
    9  h <- SHA256(x || salt)
   10  m <- <uuid, prevHash(p), seq(p)++, ts, p, q, rho, r, R, v, h>
   11  m.sig <- ECDSA-Sign(sk_PDP, m) ; LedgerQueue.enqueue(m)      (async)
   12  UpdateBaseline(p, x) if v != DENY ; return <v, sigma, m>

Every stage is timed; the response carries the per-stage milliseconds that
``scripts/latency_bench.py`` aggregates into Fig. 5.

Run locally:  ZTB_MODELS=models uvicorn ztb.pdp.app:app --port 8000
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from ztb.features.baseline import BaselineStore
from ztb.pdp import credentials
from ztb.pdp.chain import EvidenceStore, HashChain, feature_digest, new_salt, verify_chain
from ztb.pdp.context import assemble_features, control_credit, update_baseline
from ztb.pdp.queue import JsonlLedger, LedgerQueue
from ztb.pdp.schemas import AuthorizeRequest, AuthorizeResponse, RiskDetail
from ztb.pdp.settings import Settings
from ztb.pdp.signer import Signer, Verifier
from ztb.pdp.static_policy import StaticPolicy
from ztb.risk.fusion import RiskEngine
from ztb.risk.trust import band_lookup, effective_risk


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self.t0) * 1000, 3)
        self.t0 = now

    def total(self, start: float) -> None:
        self.marks["total"] = round((time.perf_counter() - start) * 1000, 3)


class PDP:
    """All state behind the endpoint; built once at startup."""

    def __init__(self, settings: Settings, engine: RiskEngine | None = None):
        self.settings = settings
        self.policy = StaticPolicy.load(settings.policy_path)
        self.engine = engine or RiskEngine.load(settings.models_dir, settings.seed,
                                                settings.device)
        self.signer = Signer.load_or_create(settings.keys_dir)
        self.verifier = Verifier.from_signer(self.signer)
        self.store = BaselineStore()
        self.chain = HashChain()
        self.ledger = JsonlLedger(settings.ledger_path)
        self.chain.rebuild(self.ledger.read_all())          # resume heads after restart
        self.queue = LedgerQueue(self.ledger)
        key = EvidenceStore.load_or_create_key(settings.keys_dir / "evidence.key")
        self.evidence = EvidenceStore(settings.evidence_path, key)
        self.alpha = engine.alpha if engine else settings.alpha

    # --- step-up challenges (line 8) ------------------------------------------

    def issue_challenge(self, principal: str, action: str, resource: str) -> str:
        exp = datetime.now(UTC) + timedelta(seconds=self.settings.challenge_ttl_seconds)
        body = f"{principal}|{action}|{resource}|{exp.isoformat()}|{uuid.uuid4().hex}"
        return body + "." + self.signer.sign_bytes(body.encode())

    def challenge_valid(self, token: str, principal: str, action: str, resource: str) -> bool:
        try:
            body, sig = token.rsplit(".", 1)
            p, a, r, exp, _ = body.split("|")
        except ValueError:
            return False
        return (self.verifier.verify_bytes(body.encode(), sig)
                and (p, a, r) == (principal, action, resource)
                and datetime.fromisoformat(exp) > datetime.now(UTC))

    # --- Algorithm 1 ---------------------------------------------------------

    def authorize(self, req: AuthorizeRequest) -> AuthorizeResponse:
        start = time.perf_counter()
        t = Timer()
        now = datetime.now(UTC)
        ctx = req.context
        sens = self.policy.sensitivity(req.resource)

        # 1. static entitlement -- evaluated first and independently
        allowed, why = self.policy.permitted(req.principal, req.action, req.resource,
                                             ctx.model_dump())
        t.lap("static_policy")
        if not allowed:
            record = self._record(req, now, r=None, R=None, verdict="DENY", x=None)
            t.lap("record_sign_enqueue")
            t.total(start)
            return AuthorizeResponse(
                verdict="DENY", reason=f"static policy: {why}", ttl_minutes=None, scope=None,
                credential=None, record=record,
                risk=RiskDetail(e=None, s=None, r=None, R=None, sensitivity=sens, credit=0.0),
                timings_ms=t.marks)

        # 2. features
        x, event = assemble_features(req, self.store, self.policy.org(req.principal), sens,
                                     now)
        t.lap("feature_assembly")

        # 3-4. detectors and fusion
        sc = self.engine.score(x[None, :], alpha=self.alpha)
        e, s, r = float(sc.e[0]), float(sc.s[0]), float(sc.r[0])
        t.lap("inference")

        # 5-6. trust and band. A valid step-up response counts as a fresh hardware MFA.
        if ctx.step_up_token and self.challenge_valid(ctx.step_up_token, req.principal,
                                                      req.action, req.resource):
            ctx = ctx.model_copy(update={"mfa_age_seconds": 0.0, "mfa_hardware_backed": True,
                                         "device_managed": True})
        c = control_credit(ctx)
        R = float(effective_risk(r, sens, c))
        band = band_lookup(R)
        verdict = band.verdict
        t.lap("trust")

        # 7-8. credential or challenge
        cred, challenge, reason = None, None, f"R={R:.3f} in band {verdict}"
        if verdict in ("ALLOW", "ALLOW_OBSERVE"):
            cred = credentials.issue(self.settings, self.signer, req.principal, req.action,
                                     req.resource, band.scope, band.ttl_minutes)
        elif verdict == "STEPUP":
            challenge = self.issue_challenge(req.principal, req.action, req.resource)
            reason += "; present step_up_token after a fresh MFA"
        t.lap("credential")

        # 9-11. digest, record, sign, enqueue
        record = self._record(req, now, r=r, R=R, verdict=verdict, x=x)
        t.lap("record_sign_enqueue")

        # 12. baseline update unless denied
        if verdict != "DENY":
            update_baseline(event, self.store)
        t.lap("baseline_update")
        t.total(start)

        return AuthorizeResponse(
            verdict=verdict, reason=reason, ttl_minutes=band.ttl_minutes,
            scope=None if verdict == "DENY" else band.scope, credential=cred,
            challenge=challenge, record=record,
            risk=RiskDetail(e=e, s=s, r=r, R=R, sensitivity=sens, credit=c),
            timings_ms=t.marks)

    def _record(self, req: AuthorizeRequest, now: datetime, *, r: float | None,
                R: float | None, verdict: str, x: np.ndarray | None) -> dict[str, Any]:
        salt = new_salt()
        seq, prev = self.chain.next(req.principal)
        record: dict[str, Any] = {
            "recId": str(uuid.uuid4()),
            "prevHash": prev,
            "seq": seq,
            "ts": now.isoformat(),
            "principal": req.principal,
            "action": req.action,
            "resource": req.resource,
            "r": None if r is None else round(r, 8),
            "R": None if R is None else round(R, 8),
            "verdict": verdict,
            "h_feat": feature_digest(x, salt),
        }
        record["sig"] = self.signer.sign_record(record)
        self.chain.advance(req.principal, record)
        self.evidence.put(record["recId"], x, salt)
        self.queue.enqueue(record)
        return record


def create_app(settings: Settings | None = None, engine: RiskEngine | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pdp = PDP(settings, engine)
        await app.state.pdp.queue.start()
        yield
        await app.state.pdp.queue.stop()

    app = FastAPI(title="ZTBAudit PDP", lifespan=lifespan)

    @app.post("/authorize", response_model=AuthorizeResponse)
    async def authorize(req: AuthorizeRequest) -> AuthorizeResponse:
        return app.state.pdp.authorize(req)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        pdp: PDP = app.state.pdp
        return {"status": "ok", "mode": settings.mode, "alpha": pdp.alpha,
                "principals_seen": len(pdp.store), "ledger_committed": pdp.queue.committed,
                "ledger_pending": pdp.queue.pending}

    @app.get("/records/{principal}")
    async def records(principal: str) -> list[dict[str, Any]]:
        pdp: PDP = app.state.pdp
        await pdp.queue.flush()
        return pdp.ledger.by_principal(principal)

    @app.get("/verify/{principal}")
    async def verify(principal: str) -> dict[str, Any]:
        pdp: PDP = app.state.pdp
        await pdp.queue.flush()
        recs = pdp.ledger.by_principal(principal)
        if not recs:
            raise HTTPException(404, f"no records for {principal}")
        first_bad = verify_chain(recs, pdp.verifier)
        return {"principal": principal, "records": len(recs), "intact": first_bad == -1,
                "first_discontinuity": first_bad}

    @app.get("/public-key")
    async def public_key() -> dict[str, str]:
        return {"pem": app.state.pdp.signer.public_pem().decode()}

    return app


app = create_app()
