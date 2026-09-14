#!/usr/bin/env python
"""Pre-demo check: venv, models, scenario, Docker, Fabric network, chaincode, shim.

Exit 0 when the demo can run (live or sim) and prints what mode it will use.
    make demo-check
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import MODELS, ROOT  # noqa: E402

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def sh(cmd: list[str], timeout: float = 15) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def main() -> int:
    hard_fail = False
    live = True
    rows: list[tuple[str, str]] = []

    def req(ok: bool, label: str, hint: str = "") -> None:
        nonlocal hard_fail
        rows.append((OK if ok else FAIL, label + ("" if ok else f" — {hint}")))
        hard_fail |= not ok

    def opt(ok: bool, label: str, hint: str = "") -> None:
        nonlocal live
        rows.append((OK if ok else WARN, label + ("" if ok else f" — {hint}")))
        live &= ok

    # --- required for any demo ---------------------------------------------------------
    req(sys.version_info[:2] == (3, 11), f"python {sys.version.split()[0]}", "needs 3.11")
    try:
        import shap  # noqa: F401
        import streamlit  # noqa: F401

        req(True, "streamlit + shap importable")
    except ImportError as e:
        req(False, "streamlit + shap importable", f"make setup ({e})")
    req((MODELS / "engine_seed0.json").exists(), "request-level engine models/engine_seed0.json",
        "make train")
    ud = MODELS / "userday"
    req((ud / "random_forest_seed0.joblib").exists() and (ud / "engine_seed0.json").exists(),
        "user-day RF + engine models/userday/",
        "python scripts/userday.py --save-models models/userday")
    meta = ROOT / "demo" / "scenario_meta.json"
    if meta.exists():
        p = json.loads(meta.read_text())["principal"]
        req((ROOT / "demo" / f"scenario_{p}.parquet").exists(), f"demo scenario {p}",
            "make demo-scenario")
    else:
        req(False, "demo scenario", "make demo-scenario")

    # --- live ledger (optional) -----------------------------------------------------------
    docker = shutil.which("docker") or str(Path.home() / ".docker/bin/docker")
    rc, _ = sh([docker, "info"])
    opt(rc == 0, "Docker daemon", "start Docker Desktop")
    if rc == 0:
        rc, out = sh([docker, "ps", "--format", "{{.Names}}"])
        names = set(out.split())
        for c in ("orderer.example.com", "peer0.org1.example.com", "peer0.org2.example.com",
                  "couchdb0"):
            opt(c in names, f"container {c}", "chaincode/README.md: network.sh up")
        opt(any("auditcontract_ccaas" in n for n in names), "chaincode-as-a-service containers",
            "deployCCAAS")
        rc, env = sh([docker, "exec", "peer0.org1.example.com", "sh", "-c",
                      "env | grep CACHESIZE"])
        opt("CACHESIZE=0" in env, "peer state cache disabled",
            "recreate peers with CORE_LEDGER_STATE_COUCHDBCONFIG_CACHESIZE=0")
    try:
        with urllib.request.urlopen("http://127.0.0.1:7071/health", timeout=3) as r:
            h = json.loads(r.read())
        opt(h.get("status") == "ok", f"gateway shim ({h.get('channel')}/{h.get('chaincode')})")
        body = json.dumps({"principal": "nobody"}).encode()
        r2 = urllib.request.Request("http://127.0.0.1:7071/VerifyChain", data=body,
                                    headers={"content-type": "application/json"})
        with urllib.request.urlopen(r2, timeout=60) as r:
            v = json.loads(r.read())
        opt(v.get("intact") is True, "VerifyChain answers on the live chain")
    except Exception as e:  # noqa: BLE001
        opt(False, "gateway shim on :7071", f"cd ztb/ledger/shim && node server.js ({e})")
    try:
        from ztb.ledger.couchdb import CouchDBAdmin

        opt(CouchDBAdmin().reachable(), "CouchDB admin access for Cover-tracks",
            "couchdb0 on :5984")
    except Exception as e:  # noqa: BLE001
        opt(False, "CouchDB admin access", str(e))

    for mark, label in rows:
        print(f"  {mark} {label}")
    if hard_fail:
        print("\nDEMO CANNOT RUN — fix the ❌ items above")
        return 1
    print(f"\nDEMO READY — ledger mode: {'LIVE FABRIC' if live else 'sim (reference ledger)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
