"""Per-principal hash chain, feature digest and off-chain evidence store (Section IV-E).

* ``seq`` is a monotonically increasing per-principal counter and ``prevHash`` the
  SHA-256 of that principal's previous record, so a suppressed record leaves a
  detectable gap even though ledger commitment is asynchronous.
* ``h_feat = SHA-256(x || salt)`` binds the feature vector to the record without
  putting it on-chain. The raw vector and salt go to an encrypted off-chain store
  (local mode: Fernet-encrypted JSONL; cloud mode: S3 Object Lock), so an auditor with
  the key can recompute ``h_feat`` and re-execute the model, while one without it
  learns nothing about the tenant's traffic from the ledger.
* :func:`verify_chain` recomputes the chain and returns the position of the first
  discontinuity, or -1 -- what the ``VerifyChain`` chaincode does in Module 4.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cryptography.fernet import Fernet

from sentinel.pdp.signer import canonical

GENESIS = "0" * 64


def record_hash(record: dict[str, Any]) -> str:
    """SHA-256 over the signed tuple plus the signature: what the next prevHash cites."""
    h = hashlib.sha256(canonical(record))
    h.update((record.get("sig") or "").encode())
    return h.hexdigest()


def feature_digest(x: np.ndarray | None, salt: bytes) -> str:
    """h_feat = SHA-256(x || salt). ``x`` is serialised as little-endian float32."""
    h = hashlib.sha256()
    if x is not None:
        h.update(np.ascontiguousarray(x, dtype="<f4").tobytes())
    h.update(salt)
    return h.hexdigest()


def new_salt() -> bytes:
    return os.urandom(16)


@dataclass
class ChainHead:
    seq: int = 0
    prev_hash: str = GENESIS


class HashChain:
    """In-memory per-principal heads. The ledger is the durable copy."""

    def __init__(self) -> None:
        self._heads: dict[str, ChainHead] = {}
        self._lock = threading.Lock()

    def head(self, principal: str) -> ChainHead:
        return self._heads.setdefault(principal, ChainHead())

    def next(self, principal: str) -> tuple[int, str]:
        """Reserve the next (seq, prevHash) for a principal."""
        with self._lock:
            h = self.head(principal)
            h.seq += 1
            return h.seq, h.prev_hash

    def advance(self, principal: str, record: dict[str, Any]) -> None:
        with self._lock:
            self.head(principal).prev_hash = record_hash(record)

    def rebuild(self, records: list[dict[str, Any]]) -> None:
        """Restore heads from a ledger on restart."""
        with self._lock:
            self._heads.clear()
            for rec in records:
                h = self.head(rec["principal"])
                h.seq = max(h.seq, int(rec["seq"]))
                h.prev_hash = record_hash(rec)


def verify_chain(records: list[dict[str, Any]], verifier=None) -> int:
    """Index of the first discontinuity for ONE principal's records in seq order; -1 if intact.

    Checks: seq starts at 1 and increments by one, prevHash equals the hash of the
    previous record, and (if a verifier is given) every signature is valid.
    """
    prev = GENESIS
    for i, rec in enumerate(records):
        if int(rec.get("seq", -1)) != i + 1:
            return i
        if rec.get("prevHash") != prev:
            return i
        if verifier is not None and not verifier.verify_record(rec):
            return i
        prev = record_hash(rec)
    return -1


class EvidenceStore:
    """Append-only, per-line Fernet-encrypted JSONL of {recId, x, salt}."""

    def __init__(self, path: Path, key: bytes):
        self.path = path
        self._f = Fernet(key)
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_or_create_key(path: Path) -> bytes:
        if path.exists():
            return path.read_bytes().strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        key = Fernet.generate_key()
        path.write_bytes(key)
        path.chmod(0o600)
        return key

    def put(self, rec_id: str, x: np.ndarray | None, salt: bytes) -> None:
        payload = json.dumps({
            "recId": rec_id,
            "x": None if x is None else np.asarray(x, dtype=np.float32).tolist(),
            "salt": salt.hex(),
        }).encode()
        line = self._f.encrypt(payload)
        with self._lock, self.path.open("ab") as fh:
            fh.write(line + b"\n")

    def get(self, rec_id: str) -> tuple[np.ndarray | None, bytes] | None:
        if not self.path.exists():
            return None
        with self.path.open("rb") as fh:
            for line in fh:
                d = json.loads(self._f.decrypt(line.strip()))
                if d["recId"] == rec_id:
                    x = None if d["x"] is None else np.asarray(d["x"], dtype=np.float32)
                    return x, bytes.fromhex(d["salt"])
        return None

    def recompute(self, rec_id: str) -> str | None:
        """h_feat from the stored evidence, for the auditor's third verification step."""
        got = self.get(rec_id)
        return None if got is None else feature_digest(*got)
