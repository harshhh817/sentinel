"""PDP runtime settings. Local mode is the default; nothing here needs AWS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from sentinel.config import FUSION_ALPHA, MODELS, ROOT

DEFAULT_POLICY = Path(__file__).parent / "policies" / "default.json"


@dataclass
class Settings:
    mode: str = field(default_factory=lambda: os.environ.get("SENTINEL_MODE", "local"))
    models_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SENTINEL_MODELS", MODELS)))
    seed: int = field(default_factory=lambda: int(os.environ.get("SENTINEL_SEED", "0")))
    alpha: float = FUSION_ALPHA
    policy_path: Path = field(
        default_factory=lambda: Path(os.environ.get("SENTINEL_POLICY", DEFAULT_POLICY)))
    state_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SENTINEL_STATE", ROOT / "state")))
    device: str = field(default_factory=lambda: os.environ.get("SENTINEL_DEVICE", "cpu"))
    # Ledger backend: jsonl (append-only file, Module 3), sim (in-process reference
    # implementation of the chaincode rules), fabric (Hyperledger Fabric via the shim).
    ledger: str = field(default_factory=lambda: os.environ.get("SENTINEL_LEDGER", "jsonl"))
    fabric_shim_url: str = field(
        default_factory=lambda: os.environ.get("SENTINEL_FABRIC_SHIM", "http://127.0.0.1:7071"))
    # Step-up challenges are valid this long; a retry after that is a fresh request.
    challenge_ttl_seconds: int = 300
    # Cloud mode only: the role the PDP assumes on the subject's behalf.
    sts_role_arn: str = field(default_factory=lambda: os.environ.get("SENTINEL_STS_ROLE_ARN", ""))

    @property
    def keys_dir(self) -> Path:
        return self.state_dir / "keys"

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / "ledger" / "local_ledger.jsonl"

    @property
    def evidence_path(self) -> Path:
        return self.state_dir / "evidence" / "evidence.enc.jsonl"

    @property
    def cloud(self) -> bool:
        return self.mode == "cloud"
