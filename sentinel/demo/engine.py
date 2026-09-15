"""Replay engine behind the dashboard: scores, explanations, two logs, cover-tracks.

Kept free of Streamlit so it is unit-testable. The right-hand "Sentinel" side reuses the
real components: the request-level RiskEngine, the user-day models (the random forest is
the operating configuration; SHAP explains it), the PDP signer and hash chain, and the
reference ledger. The left-hand "Plain IAM" side is a list that accepts anything.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sentinel.config import MODELS, ROOT
from sentinel.features.builder import FEATURE_NAMES
from sentinel.ledger.sim import SimLedger
from sentinel.pdp.chain import GENESIS, HashChain, feature_digest, new_salt
from sentinel.pdp.signer import Signer, Verifier
from sentinel.risk.fusion import RiskEngine, load_engine
from sentinel.risk.trust import band_lookup, effective_risk
from sentinel.risk.types import event_type
from sentinel.risk.userday import USERDAY_FEATURES, aggregate

_ROOT = ROOT
DEMO_DIR = ROOT / "demo"
VERDICT_COLOUR = {"ALLOW": "#2e7d32", "ALLOW_OBSERVE": "#f9a825", "STEPUP": "#ef6c00",
                  "DENY": "#c62828"}


# --- artefacts ---------------------------------------------------------------------


@dataclass
class Artefacts:
    request_engine: Any
    userday_engine: RiskEngine | None
    rf: Any
    feature_names: tuple[str, ...]
    explainer: Any = None

    @classmethod
    def load(cls, models_dir: Path = MODELS, seed: int = 0) -> Artefacts:
        req = load_engine(models_dir, seed)
        ud_dir = models_dir / "userday"
        ud_engine = rf = explainer = None
        names = USERDAY_FEATURES
        if (ud_dir / f"engine_seed{seed}.json").exists():
            ud_engine = RiskEngine.load(ud_dir, seed)
            names = tuple(json.loads((ud_dir / "userday_features.json").read_text()))
        if (ud_dir / f"random_forest_seed{seed}.joblib").exists():
            rf = joblib.load(ud_dir / f"random_forest_seed{seed}.joblib")
            import shap

            explainer = shap.TreeExplainer(rf)
        return cls(req, ud_engine, rf, names, explainer)


# --- scenario ------------------------------------------------------------------------


@dataclass
class Scenario:
    principal: str
    events: pd.DataFrame           # time-ordered, with the 34 features
    userdays: pd.DataFrame         # one row per day: x (75), label, n_events
    days: list[str]
    synthetic: bool = False

    @classmethod
    def load(cls, demo_dir: Path = DEMO_DIR, principal: str | None = None) -> Scenario:
        meta = json.loads((demo_dir / "scenario_meta.json").read_text())
        principal = principal or meta["principal"]
        path = demo_dir / f"scenario_{principal}.parquet"
        events = pq.read_table(path).to_pandas().sort_values("ts").reset_index(drop=True)
        events["day"] = events["ts"].dt.strftime("%Y-%m-%d")
        events["type"] = event_type(events["action"].to_numpy(), events["source"].to_numpy())
        ud = aggregate(path)
        userdays = pd.DataFrame(ud.x, columns=list(USERDAY_FEATURES))
        userdays.insert(0, "n_events", ud.n_events)
        userdays.insert(0, "label", ud.y)
        userdays.insert(0, "day", ud.day)
        userdays = userdays.sort_values("day").reset_index(drop=True)
        return cls(principal, events, userdays, sorted(events["day"].unique().tolist()),
                   bool(meta.get("synthetic", False)))

    def x_request(self, idx) -> np.ndarray:
        return self.events.loc[idx, list(FEATURE_NAMES)].to_numpy(np.float32)

    def x_day(self, day: str) -> np.ndarray | None:
        row = self.userdays[self.userdays["day"] == day]
        return None if row.empty else row[list(USERDAY_FEATURES)].to_numpy(np.float32)[0]


# --- scoring and explanation ---------------------------------------------------------


def day_risk(art: Artefacts, x_day: np.ndarray | None) -> float:
    """Operating configuration: the user-day random forest's P(malicious)."""
    if art.rf is None or x_day is None:
        return 0.0
    xs = art.userday_engine.standardise(x_day[None, :]) if art.userday_engine else x_day[None, :]
    return float(art.rf.predict_proba(xs)[0, 1])


def top_features(art: Artefacts, x_day: np.ndarray | None, k: int = 3) -> list[dict]:
    """Top-k SHAP contributions of the day's vector to the RF's malicious probability."""
    if art.explainer is None or x_day is None:
        return []
    xs = art.userday_engine.standardise(x_day[None, :]) if art.userday_engine else x_day[None, :]
    sv = art.explainer.shap_values(xs)
    vals = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if sv.ndim == 3 else sv[0])
    order = np.argsort(-np.abs(vals))[:k]
    return [{"feature": art.feature_names[i], "shap": float(vals[i]), "value": float(x_day[i]),
             "direction": "raises" if vals[i] > 0 else "lowers"} for i in order]


def request_scores(art: Artefacts, x: np.ndarray, types: np.ndarray, r_day: float,
                   sensitivity: np.ndarray) -> pd.DataFrame:
    sc = art.request_engine.score(x, types=types)
    r = np.maximum(sc.r, r_day)                       # propagation: max(request r, day r)
    R = effective_risk(r, sensitivity, 0.0)
    verdict = [band_lookup(v).verdict for v in R]
    return pd.DataFrame({"r_request": sc.r, "r_day": r_day, "r": r, "R": R, "verdict": verdict})


# --- the two audit trails --------------------------------------------------------------


def _ledger_backend(requested: str, shim_url: str) -> str:
    """'auto' picks the live Fabric network when the shim answers, else the reference ledger."""
    if requested != "auto":
        return requested
    try:
        from sentinel.ledger.client import FabricLedger

        FabricLedger(shim_url, timeout=3).health()
        return "fabric"
    except Exception:  # noqa: BLE001
        return "sim"


class Trails:
    """Left: a plain mutable list. Right: signed, chained records on the ledger.

    Records are enqueued and committed by an ordered background thread, exactly like the
    PDP's committer: the decision path never waits for the ledger. On Fabric each commit
    waits for its block (~2 s), so the ledger lags the replay; the dashboard shows
    committed / pending and VerifyChain covers what is committed.
    """

    def __init__(self, signer: Signer | None = None, backend: str = "auto",
                 shim_url: str = "http://127.0.0.1:7071",
                 couchdb_url: str = "http://admin:adminpw@localhost:5984/mychannel_auditcontract",
                 keys_dir: Path | None = None):
        # The channel accepts one PDP key; reuse the persisted one so the demo can talk to
        # the same network the experiments used.
        self.signer = signer or Signer.load_or_create(keys_dir or _ROOT / "state" / "keys")
        self.backend = _ledger_backend(backend, shim_url)
        self.chain = HashChain()
        self.plain: list[dict] = []
        self.tampered: list[dict] = []
        self.session = datetime.now().strftime("%H%M%S")   # fresh chains per demo session
        if self.backend == "fabric":
            from sentinel.ledger.client import FabricLedger
            from sentinel.ledger.couchdb import CouchDBAdmin

            self.ledger = FabricLedger(shim_url)
            try:
                self.ledger.set_pdp_public_key(self.signer.public_pem().decode())
            except Exception as e:  # noqa: BLE001
                if "already set" not in str(e):
                    raise
            self.couch = CouchDBAdmin(couchdb_url)
        else:
            self.ledger = SimLedger(Verifier.from_signer(self.signer))
            self.couch = None
        self._q: queue.Queue = queue.Queue()
        self.committed = 0
        self.rejected = 0
        self.last_error: str | None = None
        self._worker = threading.Thread(target=self._drain, daemon=True)
        self._worker.start()

    # --- ledger principal: the scenario user, chained per session -------------------------

    def alias(self, principal: str) -> str:
        return f"{principal}~{self.session}"

    def _drain(self) -> None:
        while True:
            rec = self._q.get()
            try:
                self.ledger.commit(rec)
                self.committed += 1
            except Exception as e:  # noqa: BLE001
                self.rejected += 1
                self.last_error = str(e)[:200]
            finally:
                self._q.task_done()

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def flush(self, timeout: float | None = None) -> None:
        if timeout is None:
            self._q.join()
            return
        end = datetime.now().timestamp() + timeout
        while self._q.unfinished_tasks and datetime.now().timestamp() < end:
            threading.Event().wait(0.1)

    # --- record / verify / tamper -----------------------------------------------------

    def record(self, principal: str, ev: pd.Series, r: float, R: float, verdict: str,
               x: np.ndarray) -> dict:
        p = self.alias(principal)
        seq, prev = self.chain.next(p)
        rec = {"recId": str(uuid.uuid4()), "prevHash": prev, "seq": seq,
               "ts": pd.Timestamp(ev["ts"]).isoformat(), "principal": p,
               "action": ev["action"], "resource": ev["resource"], "r": round(float(r), 8),
               "R": round(float(R), 8), "verdict": verdict, "h_feat": feature_digest(x, new_salt())}
        rec["sig"] = self.signer.sign_record(rec)
        self.chain.advance(p, rec)
        self._q.put(rec)                                   # async, like the PDP
        self.plain.append({**rec, "verdict": "ALLOW"})     # plain IAM logs everything as allowed
        return rec

    def ledger_records(self, principal: str) -> list[dict]:
        return self.ledger.by_principal(self.alias(principal))

    def verify(self, principal: str) -> dict:
        return self.ledger.verify_chain(self.alias(principal))

    def cover_tracks(self, principal: str, n_delete: int = 3, n_modify: int = 3,
                     rng: np.random.Generator | None = None) -> list[dict]:
        """A3 edits both stores: deletes some DENY/STEPUP records and rewrites others to ALLOW.

        On Fabric the edits go straight to the endorsing peer's CouchDB; on the reference
        ledger to its state dict. Only committed records can be tampered with.
        """
        rng = rng or np.random.default_rng(0)
        recs = self.ledger_records(principal)
        suspicious = [r for r in recs if r["verdict"] in ("DENY", "STEPUP")] or recs
        if not suspicious:
            return []
        picks = list(rng.choice(len(suspicious), size=min(n_delete + n_modify, len(suspicious)),
                                replace=False))
        actions = []
        for i, k in enumerate(picks):
            rec = suspicious[k]
            if i < n_delete:
                if self.couch is not None:
                    self.couch.delete(rec["recId"])
                else:
                    self.ledger._admin_delete(rec["recId"])
                self.plain = [p for p in self.plain if p["recId"] != rec["recId"]]
                actions.append({"op": "delete", "seq": rec["seq"], "recId": rec["recId"]})
            else:
                edited = dict(rec, verdict="ALLOW")
                if self.couch is not None:
                    self.couch.overwrite(rec["recId"], edited)
                else:
                    self.ledger._admin_overwrite(rec["recId"], edited)
                self.plain = [dict(p, verdict="ALLOW") if p["recId"] == rec["recId"] else p
                              for p in self.plain]
                actions.append({"op": "modify", "seq": rec["seq"], "recId": rec["recId"]})
        self.tampered.extend(actions)
        return actions


def genesis_ok(rec: dict) -> bool:
    return rec["prevHash"] == GENESIS if rec["seq"] == 1 else True


def ts_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
