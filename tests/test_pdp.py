"""Module 3 tests: Algorithm 1 paths, static policy, credentials, signatures, hash chain,
evidence store, committer, and the latency bench."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import InvalidToken
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sentinel.pdp import credentials  # noqa: E402
from sentinel.pdp.app import PDP, create_app  # noqa: E402
from sentinel.pdp.chain import (  # noqa: E402
    EvidenceStore,
    feature_digest,
    record_hash,
    verify_chain,
)
from sentinel.pdp.context import control_credit  # noqa: E402
from sentinel.pdp.schemas import AuthContext  # noqa: E402
from sentinel.pdp.settings import DEFAULT_POLICY, Settings  # noqa: E402
from sentinel.pdp.signer import Signer, Verifier, canonical  # noqa: E402
from sentinel.pdp.static_policy import StaticPolicy  # noqa: E402
from sentinel.risk.fusion import RiskEngine, Scores  # noqa: E402
from sentinel.risk.trust import effective_risk  # noqa: E402
from tests.fixtures import write_synthetic_splits  # noqa: E402

SCRATCH_MODELS = Path("/private/tmp/claude-501/-Users-harshgupta-projects-sentinel/"
                      "233e123e-8932-4383-b8c3-6c14df95f585/scratchpad/m2/models")


# --- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> Path:
    """Module 2 models: the synthetic-trained scratch set if present, else train a tiny one."""
    if (SCRATCH_MODELS / "engine_seed0.json").exists():
        return SCRATCH_MODELS
    from train import main as train_main

    data = tmp_path_factory.mktemp("synth")
    write_synthetic_splits(data, n_train=3000, n_val=1000, n_test=500, seed=5)
    out = tmp_path_factory.mktemp("models")
    assert train_main(["--data", str(data), "--models", str(out), "--seeds", "0",
                       "--epochs", "10", "--device", "cpu"]) == 0
    return out


class StubEngine:
    """Returns a chosen r so Algorithm 1's control flow can be driven deterministically."""

    def __init__(self, real: RiskEngine, r: float):
        self.real, self.r, self.alpha = real, r, real.alpha
        self.feature_names = real.feature_names

    def score(self, x, alpha=None):
        n = len(x)
        return Scores(np.full(n, 1.0), np.full(n, 0.5), np.full(n, self.r), np.full(n, self.r),
                      np.full(n, self.r))


def make_client(tmp_path: Path, models: Path, r: float | None = None) -> TestClient:
    settings = Settings(models_dir=models, state_dir=tmp_path / "state", device="cpu")
    engine = RiskEngine.load(models, 0)
    if r is not None:
        engine = StubEngine(engine, r)
    client = TestClient(create_app(settings, engine))
    client.__enter__()          # run lifespan: builds PDP, starts committer
    return client


def req(principal="CDE1846", action="s3:GetObject",
        resource="arn:aws:s3:::sentinel-sales/docs/q3.pdf", **ctx) -> dict:
    base = {"device_id": "PC-0001", "device_managed": True, "mfa_age_seconds": 60.0,
            "mfa_hardware_backed": True}
    base.update(ctx)
    return {"principal": principal, "action": action, "resource": resource, "context": base}


# --- Algorithm 1 paths -----------------------------------------------------


def test_allow_path_issues_scoped_credential_and_signed_record(tmp_path, models):
    with make_client(tmp_path, models, r=0.05) as c:
        body = c.post("/authorize", json=req()).json()
        assert body["verdict"] == "ALLOW" and body["ttl_minutes"] == 60
        assert body["scope"] == "as_requested" and body["challenge"] is None
        cred = body["credential"]
        assert cred["kind"] == "mock" and cred["scope"] == "as_requested"
        verifier = Verifier.from_pem(c.get("/public-key").json()["pem"].encode())
        claims = credentials.verify_mock(verifier, cred["token"])
        assert claims and claims["sub"] == "CDE1846" and claims["action"] == "s3:GetObject"
        rec = body["record"]
        assert rec["seq"] == 1 and rec["prevHash"] == "0" * 64 and rec["verdict"] == "ALLOW"
        assert verifier.verify_record(rec)
        assert set(body["timings_ms"]) >= {"static_policy", "feature_assembly", "inference",
                                           "trust", "credential", "record_sign_enqueue",
                                           "total"}


def test_allow_observe_band(tmp_path, models):
    # sens = 0.6 for the bucket, c from fresh hw MFA on managed device = 1.0
    # R = [1-(1-r)^1.9] * 0.75 ; choose r so R lands in [0.40, 0.65)
    with make_client(tmp_path, models, r=0.40) as c:
        body = c.post("/authorize", json=req()).json()
        assert 0.40 <= body["risk"]["R"] < 0.65
        assert body["verdict"] == "ALLOW_OBSERVE" and body["ttl_minutes"] == 30
        assert body["scope"] == "as_requested_verbose"


def test_stepup_path_challenge_then_retry_succeeds(tmp_path, models):
    # No credit (unmanaged device): R = 1-(1-r)^1.9 ; r = 0.5 -> R = 0.732 -> STEPUP
    with make_client(tmp_path, models, r=0.5) as c:
        first = c.post("/authorize", json=req(device_managed=False, mfa_age_seconds=None)).json()
        assert first["verdict"] == "STEPUP" and first["ttl_minutes"] == 15
        assert first["credential"] is None and first["challenge"]
        # Retry with the challenge: counts as fresh hardware MFA on a managed device -> c = 1
        retry = c.post("/authorize", json=req(device_managed=False, mfa_age_seconds=None,
                                              step_up_token=first["challenge"])).json()
        assert retry["risk"]["credit"] == 1.0
        assert retry["risk"]["R"] == pytest.approx(0.732 * 0.75, abs=1e-3)
        assert retry["verdict"] == "ALLOW_OBSERVE" and retry["credential"]
        # A forged challenge is ignored.
        forged = c.post("/authorize", json=req(device_managed=False, mfa_age_seconds=None,
                                               step_up_token="x|y|z|w|v.AAAA")).json()
        assert forged["verdict"] == "STEPUP"


def test_deny_path_no_credential_no_baseline_update_but_record_written(tmp_path, models):
    with make_client(tmp_path, models, r=0.99) as c:
        body = c.post("/authorize", json=req(device_managed=False, mfa_age_seconds=None)).json()
        assert body["verdict"] == "DENY" and body["credential"] is None
        assert body["ttl_minutes"] is None and body["scope"] is None
        pdp: PDP = c.app.state.pdp
        assert "CDE1846" not in pdp.store or pdp.store.get("CDE1846").total == 0
        recs = c.get("/records/CDE1846").json()
        assert len(recs) == 1 and recs[0]["verdict"] == "DENY" and recs[0]["R"] > 0.85


def test_static_deny_is_final_and_precedes_scoring(tmp_path, models):
    with make_client(tmp_path, models, r=0.0) as c:          # r=0 would otherwise ALLOW
        body = c.post("/authorize", json=req(resource="arn:aws:s3:::sentinel-research/x")).json()
        assert body["verdict"] == "DENY" and body["reason"].startswith("static policy")
        assert body["risk"]["r"] is None and "inference" not in body["timings_ms"]
        rec = body["record"]
        assert rec["r"] is None and rec["R"] is None and rec["verdict"] == "DENY"
        verifier = Verifier.from_pem(c.get("/public-key").json()["pem"].encode())
        assert verifier.verify_record(rec)


def test_real_engine_end_to_end_returns_a_band(tmp_path, models):
    with make_client(tmp_path, models) as c:
        for _ in range(3):
            body = c.post("/authorize", json=req()).json()
            assert body["verdict"] in {"ALLOW", "ALLOW_OBSERVE", "STEPUP", "DENY"}
            assert 0.0 <= body["risk"]["r"] <= 1.0 and 0.0 <= body["risk"]["R"] <= 1.0
        assert c.get("/health").json()["status"] == "ok"


# --- control credit (finding 2 closed) -------------------------------------


def test_control_credit_from_auth_context():
    assert control_credit(AuthContext(device_managed=False, mfa_age_seconds=0)) == 0.0
    assert control_credit(AuthContext(device_managed=True, mfa_age_seconds=None)) == 0.0
    assert control_credit(AuthContext(device_managed=True, mfa_age_seconds=0,
                                      mfa_hardware_backed=True)) == 1.0
    assert control_credit(AuthContext(device_managed=True, mfa_age_seconds=0)) == pytest.approx(0.6)
    old = control_credit(AuthContext(device_managed=True, mfa_age_seconds=9 * 3600,
                                     mfa_hardware_backed=True))
    assert old == 0.0
    # Deny stays reachable with full credit: R(r=1) = 0.75 is below 0.85, by design of
    # the credit; but any r < 1 with c = 0 can reach it.
    assert effective_risk(0.9, 1.0, 0.0) > 0.85


# --- static policy ---------------------------------------------------------


def test_static_policy_rbac_abac():
    pol = StaticPolicy.load(DEFAULT_POLICY)
    ok, _ = pol.permitted("CDE1846", "s3:GetObject", "arn:aws:s3:::sentinel-sales/x")
    assert ok
    ok, why = pol.permitted("CDE1846", "s3:GetObject", "arn:aws:s3:::sentinel-research/x")
    assert not ok and "unit bucket" in why
    ok, _ = pol.permitted("AAM0658", "s3:GetObject", "arn:aws:s3:::sentinel-research/x")
    assert ok                                                # ITAdmin: any bucket
    ok, _ = pol.permitted("CDE1846", "s3:DeleteBucket", "arn:aws:s3:::sentinel-sales")
    assert not ok                                            # deny by default
    ok, _ = pol.permitted("NOBODY", "execute-api:Invoke", "arn:aws:execute-api:::x")
    assert ok                                                # default role
    assert pol.sensitivity("arn:aws:s3:::sentinel-sales/removable/PC-1") == 1.0
    assert pol.sensitivity("arn:aws:iam::0:role/x") == 0.3


# --- credentials -----------------------------------------------------------


def test_mock_credential_expiry_and_tamper():
    signer = Signer.generate()
    v = Verifier.from_signer(signer)
    cred = credentials.issue_mock(signer, "u", "s3:GetObject", "arn:x", "read_only", 15)
    claims = credentials.verify_mock(v, cred.token)
    assert claims["scope"] == "read_only"
    assert claims["policy"]["Statement"][0]["Action"] == ["s3:GetObject", "s3:ListBucket",
                                                          "sts:GetCallerIdentity"]
    later = datetime.now(UTC) + timedelta(minutes=16)
    assert credentials.verify_mock(v, cred.token, now=later) is None
    body, sig = cred.token.rsplit(".", 1)
    assert credentials.verify_mock(v, body + "x." + sig) is None
    assert credentials.verify_mock(Verifier.from_signer(Signer.generate()), cred.token) is None


# --- signatures ------------------------------------------------------------


def _record(signer: Signer, seq=1, prev="0" * 64, verdict="ALLOW", ts="2026-01-01T00:00:00+00:00"):
    rec = {"recId": f"id-{seq}", "prevHash": prev, "seq": seq, "ts": ts, "principal": "u",
           "action": "s3:GetObject", "resource": "arn:x", "r": 0.123456789, "R": 0.2,
           "verdict": verdict, "h_feat": "ab" * 32}
    rec["sig"] = signer.sign_record(rec)
    return rec


def test_signature_rejects_modified_verdict_backdated_ts_and_foreign_key():
    signer = Signer.generate()
    v = Verifier.from_signer(signer)
    rec = _record(signer)
    assert v.verify_record(rec)
    tampered = dict(rec, verdict="DENY")
    assert not v.verify_record(tampered)
    backdated = dict(rec, ts="2025-01-01T00:00:00+00:00")
    assert not v.verify_record(backdated)
    fabricated = _record(Signer.generate())
    assert not v.verify_record(fabricated)
    assert not v.verify_record(dict(rec, sig=""))


def test_canonical_encoding_is_stable_under_float_noise_and_key_order():
    a = {"recId": "x", "r": 0.1 + 0.2, "R": 0.3, "seq": 1}
    b = {"seq": 1, "R": 0.3, "r": 0.30000000000000004, "recId": "x"}
    assert canonical(a) == canonical(b)


# --- hash chain ------------------------------------------------------------


def test_chain_continuity_detects_deletion_modification_and_fabrication():
    signer = Signer.generate()
    v = Verifier.from_signer(signer)
    recs, prev = [], "0" * 64
    for i in range(1, 8):
        r = _record(signer, seq=i, prev=prev)
        recs.append(r)
        prev = record_hash(r)
    assert verify_chain(recs, v) == -1
    deleted = recs[:3] + recs[4:]
    assert verify_chain(deleted, v) == 3
    modified = [dict(r) for r in recs]
    modified[2]["verdict"] = "DENY"
    assert verify_chain(modified, v) == 2
    fabricated = recs[:5] + [_record(Signer.generate(), seq=6, prev=record_hash(recs[4]))]
    assert verify_chain(fabricated, v) == 5
    # Without a verifier, a re-signed modification is still caught by the next prevHash.
    modified[2]["sig"] = signer.sign_record(modified[2])
    assert verify_chain(modified) == 3


def test_chain_across_requests_and_verify_endpoint(tmp_path, models):
    with make_client(tmp_path, models, r=0.05) as c:
        for _ in range(5):
            c.post("/authorize", json=req())
        c.post("/authorize",
               json=req(principal="AAM0658", resource="arn:aws:s3:::sentinel-research/y"))
        recs = c.get("/records/CDE1846").json()
        assert [r["seq"] for r in recs] == [1, 2, 3, 4, 5]
        for a, b in zip(recs, recs[1:], strict=False):
            assert b["prevHash"] == record_hash(a)
        assert c.get("/verify/CDE1846").json()["intact"] is True
        assert c.get("/verify/AAM0658").json()["records"] == 1
        # Tamper with the on-disk ledger: verify must pinpoint it.
        pdp: PDP = c.app.state.pdp
        lines = pdp.ledger.path.read_text().splitlines()
        victim = json.loads(lines[2])
        victim["verdict"] = "DENY"
        lines[2] = json.dumps(victim, sort_keys=True, separators=(",", ":"))
        pdp.ledger.path.write_text("\n".join(lines) + "\n")
        out = c.get("/verify/CDE1846").json()
        assert out["intact"] is False and out["first_discontinuity"] == victim["seq"] - 1


def test_chain_heads_survive_restart(tmp_path, models):
    settings = Settings(models_dir=models, state_dir=tmp_path / "state", device="cpu")
    engine = StubEngine(RiskEngine.load(models, 0), 0.05)
    with TestClient(create_app(settings, engine)) as c:
        c.post("/authorize", json=req())
        c.post("/authorize", json=req())
        last = c.get("/records/CDE1846").json()[-1]
    with TestClient(create_app(settings, engine)) as c:          # restart, same state dir
        body = c.post("/authorize", json=req()).json()
        assert body["record"]["seq"] == 3 and body["record"]["prevHash"] == record_hash(last)
        assert c.get("/verify/CDE1846").json()["intact"] is True


# --- evidence store ----------------------------------------------------------


def test_evidence_store_round_trip_reproduces_h_feat(tmp_path, models):
    with make_client(tmp_path, models, r=0.05) as c:
        rec = c.post("/authorize", json=req()).json()["record"]
        pdp: PDP = c.app.state.pdp
        assert pdp.evidence.recompute(rec["recId"]) == rec["h_feat"]
        x, salt = pdp.evidence.get(rec["recId"])
        assert x.shape == (34,) and len(salt) == 16
        assert feature_digest(x + 1e-3, salt) != rec["h_feat"]
        # The file on disk is ciphertext.
        raw = pdp.evidence.path.read_bytes()
        assert b"recId" not in raw and raw.startswith(b"gAAAA")


def test_evidence_wrong_key_cannot_decrypt(tmp_path):
    key = EvidenceStore.load_or_create_key(tmp_path / "k1")
    store = EvidenceStore(tmp_path / "e.jsonl", key)
    store.put("r1", np.zeros(34, np.float32), b"\x00" * 16)
    other = EvidenceStore(tmp_path / "e.jsonl", EvidenceStore.load_or_create_key(tmp_path / "k2"))
    with pytest.raises(InvalidToken):
        other.get("r1")


# --- committer ---------------------------------------------------------------


def test_committer_writes_every_record_in_order(tmp_path, models):
    with make_client(tmp_path, models, r=0.05) as c:
        ids = [c.post("/authorize", json=req()).json()["record"]["recId"] for _ in range(20)]
        recs = c.get("/records/CDE1846").json()
        assert [r["recId"] for r in recs] == ids
        assert c.get("/health").json()["ledger_committed"] == 20


# --- latency bench -----------------------------------------------------------


def test_latency_bench_writes_fig5(tmp_path, models):
    from latency_bench import STAGES
    from latency_bench import main as bench_main

    out = tmp_path / "out"
    rc = bench_main(["--models", str(models), "--n", "60", "--rps", "300", "--out", str(out),
                     "--state-dir", str(tmp_path / "st")])
    assert rc == 0
    rows = (out / "fig5.csv").read_text().splitlines()
    assert rows[0] == "stage,p50_ms,p95_ms,mean_ms" and len(rows) == len(STAGES) + 1
    meta = json.loads((out / "fig5_meta.json").read_text())
    assert meta["requests"] == 60 and meta["ledger_committed"] == 60
