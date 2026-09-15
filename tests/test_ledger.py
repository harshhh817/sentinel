"""Module 4 tests: the reference ledger's LogAccess/VerifyChain rules, the PDP committing
through it, fabricated-record rejection, the tamper harness, and the Fabric client's error
mapping."""

from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tamper_test import make_record  # noqa: E402
from tamper_test import run as tamper_run

from sentinel.ledger.client import FabricLedger  # noqa: E402
from sentinel.ledger.sim import LedgerRejected, SimLedger  # noqa: E402
from sentinel.pdp.chain import GENESIS, record_hash  # noqa: E402
from sentinel.pdp.signer import Signer, Verifier  # noqa: E402


@pytest.fixture
def pdp():
    return Signer.generate()


def chain(signer: Signer, principal: str, n: int) -> list[dict]:
    out, prev = [], GENESIS
    for s in range(1, n + 1):
        r = make_record(signer, principal, s, prev, datetime(2026, 9, 1, tzinfo=UTC))
        out.append(r)
        prev = record_hash(r)
    return out


# --- LogAccess rules -------------------------------------------------------------


def test_logaccess_accepts_a_valid_chain_and_verifychain_is_intact(pdp):
    led = SimLedger(Verifier.from_signer(pdp))
    for r in chain(pdp, "u1", 6):
        led.commit(r)
    v = led.verify_chain("u1")
    assert v["intact"] and v["records"] == 6 and v["firstDiscontinuity"] == -1
    assert [r["seq"] for r in led.by_principal("u1")] == [1, 2, 3, 4, 5, 6]
    assert led.by_principal("nobody") == []
    assert len(led.by_resource(led.by_principal("u1")[0]["resource"])) >= 1


def test_logaccess_rejects_duplicate_bad_signature_seq_gap_and_prevhash(pdp):
    led = SimLedger(Verifier.from_signer(pdp))
    recs = chain(pdp, "u1", 3)
    led.commit(recs[0])
    with pytest.raises(LedgerRejected, match="duplicate"):
        led.commit(recs[0])
    with pytest.raises(LedgerRejected, match="seq discontinuity"):
        led.commit(recs[2])                       # seq 3 before seq 2
    bad = dict(recs[1])
    bad["verdict"] = "DENY"                       # breaks the signature
    with pytest.raises(LedgerRejected, match="signature"):
        led.commit(bad)
    wrong_prev = dict(recs[1])
    wrong_prev["prevHash"] = "ff" * 32
    wrong_prev["sig"] = pdp.sign_record(wrong_prev)   # validly signed, wrong link
    with pytest.raises(LedgerRejected, match="prevHash"):
        led.commit(wrong_prev)
    led.commit(recs[1])
    assert led.verify_chain("u1")["intact"]


def test_fabricated_record_with_adversary_key_is_rejected(pdp):
    led = SimLedger(Verifier.from_signer(pdp))
    recs = chain(pdp, "u1", 2)
    for r in recs:
        led.commit(r)
    adversary = Signer.generate()
    fab = make_record(adversary, "u1", 3, record_hash(recs[1]), datetime.now(UTC))
    ok, why = led.try_commit(fab)
    assert not ok and "signature" in why
    assert led.verify_chain("u1")["records"] == 2 and led.verify_chain("u1")["intact"]


def test_verifychain_locates_deletion_modification_and_truncated_tail(pdp):
    led = SimLedger(Verifier.from_signer(pdp))
    recs = chain(pdp, "u1", 5)
    for r in recs:
        led.commit(r)
    led._admin_delete(recs[2]["recId"])          # interior deletion
    v = led.verify_chain("u1")
    assert not v["intact"] and v["firstDiscontinuity"] == 2
    led2 = SimLedger(Verifier.from_signer(pdp))
    for r in recs:
        led2.commit(r)
    m = dict(recs[1])
    m["verdict"] = "DENY"
    led2._admin_overwrite(m["recId"], m)        # in-place modification
    assert led2.verify_chain("u1")["firstDiscontinuity"] == 1
    led3 = SimLedger(Verifier.from_signer(pdp))
    for r in recs:
        led3.commit(r)
    led3._admin_delete(recs[4]["recId"])         # tail deletion: caught via the head
    v = led3.verify_chain("u1")
    assert not v["intact"] and v["firstDiscontinuity"] == 4 and "head says" in v["reason"]


# --- PDP committing through the ledger ---------------------------------------------


def test_pdp_commits_through_sim_ledger_and_counts_rejections(tmp_path):
    from fastapi.testclient import TestClient

    from sentinel.pdp.app import create_app
    from sentinel.pdp.settings import Settings
    from sentinel.risk.fusion import RiskEngine
    from tests.test_pdp import SCRATCH_MODELS, StubEngine, req

    models = SCRATCH_MODELS if (SCRATCH_MODELS / "engine_seed0.json").exists() else Path("models")
    if not (models / "engine_seed0.json").exists():
        pytest.skip("no trained engine available")
    settings = Settings(models_dir=models, state_dir=tmp_path / "state", device="cpu",
                        ledger="sim")
    engine = StubEngine(RiskEngine.load(models, 0), 0.05)
    with TestClient(create_app(settings, engine)) as c:
        for _ in range(4):
            assert c.post("/authorize", json=req()).json()["verdict"] == "ALLOW"
        v = c.get("/verify/CDE1846").json()
        assert v["backend"] == "sim" and v["intact"] and v["records"] == 4
        h = c.get("/health").json()
        assert h["ledger"] == "sim" and h["ledger_committed"] == 4 and h["ledger_rejected"] == 0
        # A forged record pushed into the queue by an attacker with queue access is rejected
        # at LogAccess and counted, and the chain stays intact.
        pdp = c.app.state.pdp
        forged = make_record(Signer.generate(), "CDE1846", 5,
                             record_hash(pdp.ledger.by_principal("CDE1846")[-1]),
                             datetime.now(UTC))
        pdp.queue.enqueue(forged)
        c.get("/records/CDE1846")                 # flushes the queue
        h = c.get("/health").json()
        assert h["ledger_rejected"] == 1 and h["ledger_committed"] == 4
        assert c.get("/verify/CDE1846").json()["intact"]


# --- tamper harness ------------------------------------------------------------------


def test_tamper_harness_detects_everything_with_no_false_alarms():
    rows, meta = tamper_run("sim", n_attempts=100, control=1000, seed=1, shim_url=None)
    assert len(rows) == 100 and meta["control_alarms"] == 0
    assert all(r["jsonl_succeeded"] and not r["jsonl_detected"] for r in rows)
    assert sum(r["ledger_detected"] for r in rows) == 100
    kinds = {r["kind"] for r in rows}
    assert kinds == {"delete", "modify", "backdate", "fabricate"}


# --- Fabric client error mapping ------------------------------------------------------


class _Shim(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/LogAccess" and body["record"].get("verdict") == "DENY":
            self.send_response(422)
            self.end_headers()
            self.wfile.write(b"signature does not verify for recId x")
            return
        if self.path == "/VerifyChain":
            out = {"principal": body["principal"], "records": 3, "intact": True,
                   "firstDiscontinuity": -1}
        else:
            out = {"ok": True}
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, *a):  # silence
        pass


def test_fabric_client_maps_422_to_ledger_rejected():
    srv = HTTPServer(("127.0.0.1", 0), _Shim)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = FabricLedger(f"http://127.0.0.1:{srv.server_port}")
        client.commit({"recId": "a", "verdict": "ALLOW"})
        with pytest.raises(LedgerRejected, match="signature"):
            client.commit({"recId": "b", "verdict": "DENY"})
        v = client.verify_chain("u1")
        assert v["intact"] and v["firstDiscontinuity"] == -1
    finally:
        srv.shutdown()
