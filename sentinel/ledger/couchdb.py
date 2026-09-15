"""Direct edits to an endorsing peer's CouchDB state database -- the adversary's handle.

On Fabric the world state is only writable through endorsed transactions, so a privileged
log manipulator (A3) with host access edits the peer's CouchDB documents behind the
chaincode's back. The tamper experiment and the demo's "Cover tracks" both use this;
``VerifyChain`` has to catch the result from the state alone.
"""

from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_URL = "http://admin:adminpw@localhost:5984/mychannel_auditcontract"


class CouchDBAdmin:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 10.0):
        u = urllib.parse.urlsplit(url)
        self.base = f"{u.scheme}://{u.hostname}:{u.port}{u.path}"
        self.headers = {"content-type": "application/json"}
        if u.username:                       # urllib does not accept user:pass@ in URLs
            token = base64.b64encode(f"{u.username}:{u.password or ''}".encode()).decode()
            self.headers["Authorization"] = f"Basic {token}"
        self.timeout = timeout

    def _url(self, rec_id: str) -> str:
        return f"{self.base}/{urllib.parse.quote('rec/' + rec_id, safe='')}"

    def _req(self, url: str, method: str = "GET", body: dict | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method, headers=self.headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None

    def reachable(self) -> bool:
        try:
            self._req(self.base)
            return True
        except Exception:  # noqa: BLE001
            return False

    def get(self, rec_id: str) -> dict:
        return self._req(self._url(rec_id))

    def delete(self, rec_id: str) -> None:
        doc = self.get(rec_id)
        self._req(f"{self._url(rec_id)}?rev={doc['_rev']}", "DELETE")

    def overwrite(self, rec_id: str, fields: dict) -> None:
        doc = self.get(rec_id)
        body = dict(doc)
        body.update({k: v for k, v in fields.items() if not k.startswith("_")})
        self._req(self._url(rec_id), "PUT", body)
