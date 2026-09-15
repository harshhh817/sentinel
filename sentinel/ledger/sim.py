"""Reference implementation of the audit chaincode's rules, in Python.

Mirrors chaincode/auditcontract exactly -- LogAccess validation (duplicate recId,
signature against the PDP key, per-principal seq continuity, prevHash), the two
queries and VerifyChain -- as an in-process :class:`~sentinel.pdp.queue.LedgerSink`. It is
what the tests and the tamper experiment run against when no Fabric network is up,
and it lets Module 3 be exercised end to end on a laptop without Docker. It is *not*
tamper-evident by itself: a Fabric ledger is what makes the rules externally
enforceable. Results produced against it are labelled ``ledger=sim``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from sentinel.pdp.chain import GENESIS, record_hash
from sentinel.pdp.signer import Verifier


class LedgerRejected(Exception):
    """What LogAccess raises when endorsement would fail."""


class SimLedger:
    def __init__(self, verifier: Verifier, path: Path | None = None):
        self.verifier = verifier
        self._records: dict[str, dict[str, Any]] = {}
        self._by_principal: dict[str, list[str]] = {}
        self._heads: dict[str, tuple[int, str]] = {}
        self._lock = threading.Lock()
        self.path = path                       # optional append-only journal, for inspection
        self.rejected: list[tuple[str, str]] = []

    # --- LogAccess -------------------------------------------------------------

    def commit(self, record: dict[str, Any]) -> None:
        """LogAccess. Raises :class:`LedgerRejected` with the chaincode's reason."""
        with self._lock:
            rec_id, principal = record.get("recId"), record.get("principal")
            if not rec_id or not principal:
                raise LedgerRejected("recId and principal are required")
            if rec_id in self._records:
                raise LedgerRejected(f"duplicate recId {rec_id}")
            if not self.verifier.verify_record(record):
                raise LedgerRejected(f"signature does not verify for recId {rec_id}")
            seq, prev = self._heads.get(principal, (0, GENESIS))
            if int(record.get("seq", -1)) != seq + 1:
                raise LedgerRejected(f"seq discontinuity for {principal}: "
                                     f"got {record.get('seq')}, expected {seq + 1}")
            if record.get("prevHash") != prev:
                raise LedgerRejected(f"prevHash mismatch for {principal} at seq {record['seq']}")
            self._records[rec_id] = dict(record)
            self._by_principal.setdefault(principal, []).append(rec_id)
            self._heads[principal] = (int(record["seq"]), record_hash(record))
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as fh:
                    fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def try_commit(self, record: dict[str, Any]) -> tuple[bool, str]:
        try:
            self.commit(record)
            return True, ""
        except LedgerRejected as e:
            self.rejected.append((record.get("recId", "?"), str(e)))
            return False, str(e)

    # --- queries -------------------------------------------------------------------

    def read_all(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records.values()]

    def by_principal(self, principal: str) -> list[dict[str, Any]]:
        return [dict(self._records[i]) for i in self._by_principal.get(principal, [])]

    def by_resource(self, resource: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records.values() if r.get("resource") == resource]

    def verify_chain(self, principal: str) -> dict[str, Any]:
        recs = self.by_principal(principal)
        prev = GENESIS
        idx, reason = -1, ""
        for i, rec in enumerate(recs):
            if int(rec.get("seq", -1)) != i + 1:
                idx, reason = i, f"seq {rec.get('seq')} at position {i}"
                break
            if rec.get("prevHash") != prev:
                idx, reason = i, f"prevHash mismatch at seq {rec['seq']}"
                break
            if not self.verifier.verify_record(rec):
                idx, reason = i, f"bad signature at seq {rec['seq']}"
                break
            prev = record_hash(rec)
        if idx == -1 and principal in self._heads and len(recs) != self._heads[principal][0]:
            # A truncated tail is internally consistent; the head written at the last
            # accepted LogAccess says how long the chain must be.
            idx = len(recs)
            reason = f"head says seq {self._heads[principal][0]} but {len(recs)} records present"
        return {"principal": principal, "records": len(recs), "intact": idx == -1,
                "firstDiscontinuity": idx, "reason": reason}

    # --- the adversary's only handle: direct state-database access -----------------

    def _admin_delete(self, rec_id: str) -> None:
        """Simulates A3 deleting from the peer's state DB behind the chaincode's back."""
        with self._lock:
            rec = self._records.pop(rec_id, None)
            if rec:
                self._by_principal[rec["principal"]].remove(rec_id)

    def _admin_overwrite(self, rec_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._records[rec_id] = dict(record)
