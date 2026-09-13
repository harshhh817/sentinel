#!/usr/bin/env python
"""Table VII tamper experiment: 500 attempts against a plain log vs the ledger.

Simulates A3, the privileged log manipulator, with administrative access to both a
mirrored conventional JSONL log and the peer's state. Attempts are evenly divided:

  delete     remove a record                          -> gap in the per-principal chain
  modify     change a record's verdict in place       -> signature no longer verifies
  backdate   rewrite a record's timestamp in place    -> signature no longer verifies
  fabricate  append a benign-looking record signed    -> rejected at endorsement
             with the adversary's own key

Against the JSONL log every attempt "succeeds" (the file accepts anything) and none is
self-evident. Against the ledger, in-place modification and fabrication are rejected
at LogAccess, and deletion / suppression is caught by VerifyChain. A control run of
10,000 untampered records must raise zero alarms.

    python scripts/tamper_test.py --ledger sim --out results        # laptop, no Docker
    python scripts/tamper_test.py --ledger fabric --out results     # live test-network
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import RESULTS  # noqa: E402
from ztb.ledger.sim import LedgerRejected, SimLedger  # noqa: E402
from ztb.pdp.chain import GENESIS, feature_digest, new_salt, record_hash, verify_chain  # noqa: E402
from ztb.pdp.signer import Signer, Verifier  # noqa: E402

KINDS = ("delete", "modify", "backdate", "fabricate")


def make_record(signer: Signer, principal: str, seq: int, prev: str, ts: datetime,
                verdict: str = "ALLOW") -> dict:
    rec = {"recId": str(uuid.uuid4()), "prevHash": prev, "seq": seq, "ts": ts.isoformat(),
           "principal": principal, "action": "s3:GetObject",
           "resource": f"arn:aws:s3:::ztb-sales/docs/{seq % 50}.pdf",
           "r": round(random.random(), 8), "R": round(random.random(), 8), "verdict": verdict,
           "h_feat": feature_digest(None, new_salt())}
    rec["sig"] = signer.sign_record(rec)
    return rec


RUN_TAG = datetime.now().strftime("%H%M%S")   # unique principals per run on a reused channel


def commit_chains(ledger, records: list[dict], workers: int = 32) -> int:
    """Commit records with per-principal ordering preserved, principals in parallel.

    Each Fabric submit waits ~2 s for block commit, so sequential submission of ten
    thousand records would take hours; chains are independent, so they run in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    by_p: dict[str, list[dict]] = {}
    for r in records:
        by_p.setdefault(r["principal"], []).append(r)
    rejected = 0

    def run_chain(chain: list[dict]) -> int:
        bad = 0
        for r in sorted(chain, key=lambda x: x["seq"]):
            try:
                ledger.commit(r)
            except LedgerRejected:
                bad += 1
        return bad

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for bad in ex.map(run_chain, by_p.values()):
            rejected += bad
    return rejected


def generate_chains(signer: Signer, principals: int, per_principal: int, rng: random.Random):
    """Honest PDP output: per-principal chains, interleaved in time."""
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    out = []
    for p in range(principals):
        prev, principal = GENESIS, f"U{RUN_TAG}-{p:04d}"
        for s in range(1, per_principal + 1):
            rec = make_record(signer, principal, s, prev,
                              t0 + timedelta(minutes=rng.randrange(60 * 24 * 30)))
            out.append(rec)
            prev = record_hash(rec)
    return out


class JsonlLog:
    """The conventional mutable log the adversary can edit freely."""

    def __init__(self):
        self.records: list[dict] = []

    def commit(self, rec):
        self.records.append(dict(rec))

    def by_principal(self, p):
        return sorted((r for r in self.records if r["principal"] == p), key=lambda r: r["seq"])

    def delete(self, rec_id):
        self.records = [r for r in self.records if r["recId"] != rec_id]

    def overwrite(self, rec):
        self.records = [rec if r["recId"] == rec["recId"] else r for r in self.records]


def run(ledger_kind: str, n_attempts: int, control: int, seed: int, shim_url: str | None,
        keys_dir: Path | None = None):
    rng = random.Random(seed)
    random.seed(seed)
    # A persisted key lets several scripts share one channel (the chaincode accepts the
    # PDP key once); without it every run needs a fresh channel.
    pdp = Signer.load_or_create(keys_dir) if keys_dir else Signer.generate()
    verifier = Verifier.from_signer(pdp)
    adversary = Signer.generate()

    if ledger_kind == "sim":
        ledger = SimLedger(verifier)
    elif ledger_kind == "fabric":
        from ztb.ledger.client import FabricLedger

        ledger = FabricLedger(shim_url) if shim_url else FabricLedger()
        try:
            ledger.set_pdp_public_key(pdp.public_pem().decode())
        except Exception as e:  # noqa: BLE001
            if "already set" not in str(e) or not keys_dir:
                raise SystemExit(f"cannot install the PDP key on the ledger: {e}") from e
    else:
        raise SystemExit(f"unknown ledger {ledger_kind}")
    log = JsonlLog()

    # --- honest history: 500 tamper attempts need targets; control needs 10k clean records
    per_attempt = n_attempts // len(KINDS)
    principals = 50
    per_principal = max(20, (n_attempts * 2) // principals)
    history = generate_chains(pdp, principals, per_principal, rng)
    for rec in history:
        log.commit(rec)
    if commit_chains(ledger, history):
        raise SystemExit("honest history was rejected by the ledger; is the channel fresh?")

    # --- 500 attempts
    rows = []
    targets = rng.sample(history, per_attempt * 3)   # distinct targets: delete/modify/backdate
    for i, kind in enumerate(KINDS):
        for j in range(per_attempt):
            if kind == "fabricate":
                # a benign-looking record for a real principal, correct seq and prevHash,
                # signed by the adversary
                p = f"U{RUN_TAG}-{rng.randrange(principals):04d}"
                chain = log.by_principal(p)
                rec = make_record(adversary, p, chain[-1]["seq"] + 1, record_hash(chain[-1]),
                                  datetime.now(UTC))
                log.commit(rec)
                log_success, log_detected = True, False
                try:
                    ledger.commit(rec)
                    ledger_success = True
                except LedgerRejected:
                    ledger_success = False
                ledger_detected = not ledger_success
                if not ledger_detected:                # if it got in, would VerifyChain see it?
                    ledger_detected = ledger.verify_chain(p)["intact"] is False
            else:
                target = targets[i * per_attempt + j]
                p = target["principal"]
                if kind == "delete":
                    log.delete(target["recId"])
                    _admin(ledger, "delete", target)
                else:
                    t = dict(target)
                    if kind == "modify":
                        t["verdict"] = "DENY" if t["verdict"] != "DENY" else "ALLOW"
                    else:
                        earlier = datetime.fromisoformat(t["ts"]) - timedelta(days=30)
                        t["ts"] = earlier.isoformat()
                    log.overwrite(t)
                    _admin(ledger, "overwrite", t)
                log_success = True
                # a plain log has nothing to check against; "detected" means self-evident
                log_detected = False
                ledger_success = True                  # the state DB edit itself succeeds ...
                ledger_detected = ledger.verify_chain(p)["intact"] is False  # ... but is caught
            rows.append({"attempt": len(rows) + 1, "kind": kind,
                         "jsonl_succeeded": log_success, "jsonl_detected": log_detected,
                         "ledger_undetected": ledger_success and not ledger_detected,
                         "ledger_detected": ledger_detected})

    # --- control: 10,000 untampered records on fresh principals, zero alarms expected
    ctrl_pdp = pdp
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    ctrl_principals = 100
    ctrl: list[dict] = []
    for p in range(ctrl_principals):
        principal, prev = f"C{RUN_TAG}-{p:04d}", GENESIS
        for s in range(1, control // ctrl_principals + 1):
            rec = make_record(ctrl_pdp, principal, s, prev, t0 + timedelta(seconds=s))
            ctrl.append(rec)
            prev = record_hash(rec)
    alarms = commit_chains(ledger, ctrl)               # any rejection of a clean record
    for p in range(ctrl_principals):
        principal = f"C{RUN_TAG}-{p:04d}"
        if not ledger.verify_chain(principal)["intact"]:
            alarms += 1
        # the Python-side chain check must agree on the clean control set
        if verify_chain(ledger.by_principal(principal), verifier) != -1:
            alarms += 1

    return rows, {"ledger": ledger_kind, "attempts": len(rows), "per_kind": per_attempt,
                  "control_records": control, "control_alarms": alarms,
                  "honest_records": len(history)}


COUCHDB = None   # set from --couchdb; e.g. http://admin:adminpw@localhost:5984/mychannel_auditcontract


def _admin(ledger, op: str, rec: dict) -> None:
    """A3's direct write to the peer's state database, bypassing the chaincode.

    Against Fabric this edits the endorsing peer's CouchDB document for ``rec/<recId>``
    directly (test-network exposes couchdb0 with admin credentials), which is exactly
    what a privileged log manipulator with host access would do; the chaincode never
    sees it, and VerifyChain has to catch it from the state alone.
    """
    if isinstance(ledger, SimLedger):
        if op == "delete":
            ledger._admin_delete(rec["recId"])
        else:
            ledger._admin_overwrite(rec["recId"], rec)
        return
    if not COUCHDB:
        raise SystemExit("--couchdb is required for fabric-mode tampering "
                         "(e.g. http://admin:adminpw@localhost:5984/mychannel_auditcontract)")
    import base64
    import urllib.parse
    import urllib.request

    # urllib does not accept user:pass@ in URLs; send basic auth as a header instead.
    u = urllib.parse.urlsplit(COUCHDB)
    headers = {"content-type": "application/json"}
    if u.username:
        token = base64.b64encode(f"{u.username}:{u.password or ''}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    base = f"{u.scheme}://{u.hostname}:{u.port}{u.path}"
    doc_url = f"{base}/{urllib.parse.quote('rec/' + rec['recId'], safe='')}"
    with urllib.request.urlopen(urllib.request.Request(doc_url, headers=headers),
                                timeout=10) as resp:
        doc = json.loads(resp.read())
    if op == "delete":
        req = urllib.request.Request(f"{doc_url}?rev={doc['_rev']}", method="DELETE",
                                     headers=headers)
    else:
        body = dict(doc)
        body.update({k: v for k, v in rec.items()})          # edited fields, same _id/_rev
        req = urllib.request.Request(doc_url, data=json.dumps(body).encode(), method="PUT",
                                     headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", choices=["sim", "fabric"], default="sim")
    ap.add_argument("--attempts", type=int, default=500)
    ap.add_argument("--control", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shim", default=None)
    ap.add_argument("--keys-dir", type=Path, default=None,
                    help="persisted PDP key (share one channel across scripts)")
    ap.add_argument("--couchdb", default=None,
                    help="fabric mode: the endorsing peer's state DB, for A3's direct edits")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    global COUCHDB
    COUCHDB = a.couchdb

    rows, meta = run(a.ledger, a.attempts, a.control, a.seed, a.shim, a.keys_dir)
    by_kind = {}
    for r in rows:
        k = by_kind.setdefault(r["kind"], {"attempts": 0, "jsonl_succeeded": 0,
                                           "jsonl_detected": 0, "ledger_detected": 0})
        k["attempts"] += 1
        k["jsonl_succeeded"] += r["jsonl_succeeded"]
        k["jsonl_detected"] += r["jsonl_detected"]
        k["ledger_detected"] += r["ledger_detected"]
    total_det = sum(k["ledger_detected"] for k in by_kind.values())

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "table_vii.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "ledger"])
        for kind, k in by_kind.items():
            n = k["attempts"]
            w.writerow([f"{kind}: succeeded against JSONL log", f"{k['jsonl_succeeded']} / {n}",
                        "jsonl"])
            w.writerow([f"{kind}: self-evident in JSONL log", f"{k['jsonl_detected']} / {n}",
                        "jsonl"])
            w.writerow([f"{kind}: detected on ledger", f"{k['ledger_detected']} / {n}", a.ledger])
        w.writerow(["Tampering attempts detected", f"{total_det} / {len(rows)}", a.ledger])
        w.writerow(["False alarms over clean records",
                    f"{meta['control_alarms']} / {meta['control_records']}", a.ledger])
    (a.out / "table_vii_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    for kind, k in by_kind.items():
        print(f"  {kind:10s} jsonl succeeded {k['jsonl_succeeded']:>3}/{k['attempts']}  "
              f"self-evident {k['jsonl_detected']:>3}   "
              f"ledger detected {k['ledger_detected']:>3}/{k['attempts']}")
    print(f"  TOTAL detected {total_det}/{len(rows)}   "
          f"control alarms {meta['control_alarms']}/{meta['control_records']}")
    print(f"wrote {a.out / 'table_vii.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
