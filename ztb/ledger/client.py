"""Fabric ledger client for the Module 3 committer.

The Python Fabric SDKs lag the Gateway API, so the connection lives in a tiny Node
service (``ztb/ledger/shim/``) built on the official ``@hyperledger/fabric-gateway``
package. It exposes the four chaincode transactions over local HTTP/JSON; this client
is a thin, dependency-free wrapper that implements :class:`~ztb.pdp.queue.LedgerSink`.

    node ztb/ledger/shim/server.js          # after ./network.sh up + deployCC
    ZTB_LEDGER=fabric uvicorn ztb.pdp.app:app

Commit failures surface as :class:`~ztb.ledger.sim.LedgerRejected` with the
chaincode's message, so endorsement rejections (bad signature, seq gap, duplicate)
are distinguishable from transport errors.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from ztb.ledger.sim import LedgerRejected

DEFAULT_URL = os.environ.get("ZTB_FABRIC_SHIM", "http://127.0.0.1:7071")


class FabricLedger:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 150.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _call(self, path: str, payload: dict[str, Any] | None = None,
              retries: int = 3) -> Any:
        data = None if payload is None else json.dumps(payload).encode()
        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(self.url + path, data=data,
                                         headers={"content-type": "application/json"},
                                         method="POST" if data is not None else "GET")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read() or b"null")
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code == 422:                 # endorsement rejected by the chaincode
                    raise LedgerRejected(body) from None
                last = RuntimeError(f"fabric shim {path}: HTTP {e.code}: {body}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = RuntimeError(f"fabric shim {path}: {e}")
            time.sleep(1.5 * (attempt + 1))      # transport/deadline errors are retried
        raise last  # type: ignore[misc]

    # --- LedgerSink -------------------------------------------------------------

    def commit(self, record: dict[str, Any]) -> None:
        self._call("/LogAccess", {"record": record})

    def read_all(self) -> list[dict[str, Any]]:
        return self._call("/all")

    def by_principal(self, principal: str) -> list[dict[str, Any]]:
        return self._call("/QueryByPrincipal", {"principal": principal})

    def by_resource(self, resource: str) -> list[dict[str, Any]]:
        return self._call("/QueryByResource", {"resource": resource})

    def verify_chain(self, principal: str) -> dict[str, Any]:
        return self._call("/VerifyChain", {"principal": principal})

    def set_pdp_public_key(self, pem: str) -> None:
        self._call("/SetPDPPublicKey", {"pem": pem})

    def health(self) -> dict[str, Any]:
        return self._call("/health")
