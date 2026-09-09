"""ECDSA P-256 signing of audit records (eq. 5, Section IV-E).

The signature covers the canonical JSON of the record tuple *without* ``sig``:
sorted keys, no whitespace, floats rounded to 8 places so the same tuple always
serialises to the same bytes. Verification recomputes that encoding, which is why
a modified verdict or backdated timestamp fails at endorsement.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

SIGNED_FIELDS = ("recId", "prevHash", "seq", "ts", "principal", "action", "resource",
                 "r", "R", "verdict", "h_feat")


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 8)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_norm(x) for x in v]
    return v


def canonical(record: dict[str, Any], fields: tuple[str, ...] = SIGNED_FIELDS) -> bytes:
    """Deterministic bytes of the signed tuple."""
    payload = {k: _norm(record.get(k)) for k in fields}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


class Signer:
    def __init__(self, private_key: ec.EllipticCurvePrivateKey):
        self._sk = private_key
        self.public_key = private_key.public_key()

    @classmethod
    def generate(cls) -> Signer:
        return cls(ec.generate_private_key(ec.SECP256R1()))

    @classmethod
    def load_or_create(cls, keys_dir: Path) -> Signer:
        keys_dir.mkdir(parents=True, exist_ok=True)
        sk_path = keys_dir / "pdp_p256.pem"
        if sk_path.exists():
            sk = serialization.load_pem_private_key(sk_path.read_bytes(), password=None)
            return cls(sk)
        signer = cls.generate()
        sk_path.write_bytes(signer._sk.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
        sk_path.chmod(0o600)
        (keys_dir / "pdp_p256.pub.pem").write_bytes(signer.public_pem())
        return signer

    def public_pem(self) -> bytes:
        return self.public_key.public_bytes(serialization.Encoding.PEM,
                                            serialization.PublicFormat.SubjectPublicKeyInfo)

    def sign_bytes(self, data: bytes) -> str:
        return base64.b64encode(self._sk.sign(data, ec.ECDSA(hashes.SHA256()))).decode()

    def sign_record(self, record: dict[str, Any]) -> str:
        return self.sign_bytes(canonical(record))


class Verifier:
    def __init__(self, public_key: ec.EllipticCurvePublicKey):
        self._pk = public_key

    @classmethod
    def from_pem(cls, pem: bytes) -> Verifier:
        return cls(serialization.load_pem_public_key(pem))

    @classmethod
    def from_signer(cls, signer: Signer) -> Verifier:
        return cls(signer.public_key)

    def verify_bytes(self, data: bytes, sig_b64: str) -> bool:
        try:
            self._pk.verify(base64.b64decode(sig_b64), data, ec.ECDSA(hashes.SHA256()))
            return True
        except (InvalidSignature, ValueError):
            return False

    def verify_record(self, record: dict[str, Any]) -> bool:
        sig = record.get("sig")
        return bool(sig) and self.verify_bytes(canonical(record), sig)
