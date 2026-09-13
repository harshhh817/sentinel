"""Replay engine behind the dashboard: scores, explanations, two logs, cover-tracks.

Kept free of Streamlit so it is unit-testable. The right-hand "ZTBAudit" side reuses the
real components: the request-level RiskEngine, the user-day models (the random forest is
the operating configuration; SHAP explains it), the PDP signer and hash chain, and the
reference ledger. The left-hand "Plain IAM" side is a list that accepts anything.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ztb.config import MODELS, ROOT
from ztb.features.builder import FEATURE_NAMES
from ztb.ledger.sim import SimLedger
from ztb.pdp.chain import GENESIS, HashChain, feature_digest, new_salt
from ztb.pdp.signer import Signer, Verifier
from ztb.risk.fusion import RiskEngine, load_engine
from ztb.risk.trust import band_lookup, effective_risk
from ztb.risk.types import event_type
from ztb.risk.userday import USERDAY_FEATURES, aggregate

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
        return cls(principal, events, userdays, sorted(events["day"].unique().tolist()))

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


@dataclass
class Trails:
    """Left: a plain mutable list. Right: signed, chained records on the reference ledger."""

    signer: Signer
    chain: HashChain = field(default_factory=HashChain)
    ledger: SimLedger = field(init=False)
    plain: list[dict] = field(default_factory=list)
    tampered: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.ledger = SimLedger(Verifier.from_signer(self.signer))

    def record(self, principal: str, ev: pd.Series, r: float, R: float, verdict: str,
               x: np.ndarray) -> dict:
        seq, prev = self.chain.next(principal)
        rec = {"recId": str(uuid.uuid4()), "prevHash": prev, "seq": seq,
               "ts": pd.Timestamp(ev["ts"]).isoformat(), "principal": principal,
               "action": ev["action"], "resource": ev["resource"], "r": round(float(r), 8),
               "R": round(float(R), 8), "verdict": verdict, "h_feat": feature_digest(x, new_salt())}
        rec["sig"] = self.signer.sign_record(rec)
        self.chain.advance(principal, rec)
        self.ledger.commit(rec)
        self.plain.append({**rec, "verdict": "ALLOW"})   # plain IAM logs everything as allowed
        return rec

    def verify(self, principal: str) -> dict:
        return self.ledger.verify_chain(principal)

    def cover_tracks(self, principal: str, n_delete: int = 3, n_modify: int = 3,
                     rng: np.random.Generator | None = None) -> list[dict]:
        """A3 edits both stores: deletes some DENY/STEPUP records and rewrites others to ALLOW."""
        rng = rng or np.random.default_rng(0)
        recs = self.ledger.by_principal(principal)
        suspicious = [r for r in recs if r["verdict"] in ("DENY", "STEPUP")] or recs
        picks = list(rng.choice(len(suspicious), size=min(n_delete + n_modify, len(suspicious)),
                                replace=False))
        actions = []
        for i, k in enumerate(picks):
            rec = suspicious[k]
            if i < n_delete:
                self.ledger._admin_delete(rec["recId"])
                self.plain = [p for p in self.plain if p["recId"] != rec["recId"]]
                actions.append({"op": "delete", "seq": rec["seq"], "recId": rec["recId"]})
            else:
                edited = dict(rec, verdict="ALLOW")
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
