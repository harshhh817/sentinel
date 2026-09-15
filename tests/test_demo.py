"""Module 3b tests: replay engine, SHAP top-3, the two trails, and cover-tracks detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from sentinel.demo.engine import (
    DEMO_DIR,
    Artefacts,
    Scenario,
    Trails,
    day_risk,
    request_scores,
    top_features,
)
from sentinel.pdp.signer import Signer
from sentinel.risk.userday import USERDAY_FEATURES

HAVE_SCENARIO = (DEMO_DIR / "scenario_meta.json").exists()
HAVE_MODELS = (Path("models") / "engine_seed0.json").exists()


@pytest.fixture
def tiny_art():
    """Artefacts with a tiny RF + SHAP explainer trained in-test; no request engine."""
    import shap

    rng = np.random.default_rng(0)
    n = len(USERDAY_FEATURES)
    x = rng.normal(size=(400, n)).astype(np.float32)
    y = (x[:, USERDAY_FEATURES.index("after_hours_frac")] + 0.5 * x[:, 0] > 1.0).astype(int)
    rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=0).fit(x, y)
    return Artefacts(request_engine=None, userday_engine=None, rf=rf,
                     feature_names=USERDAY_FEATURES, explainer=shap.TreeExplainer(rf))


def test_top_features_returns_three_named_contributions(tiny_art):
    x = np.zeros(len(USERDAY_FEATURES), np.float32)
    x[USERDAY_FEATURES.index("after_hours_frac")] = 3.0
    x[0] = 2.0
    top = top_features(tiny_art, x)
    assert len(top) == 3
    assert {t["feature"] for t in top} <= set(USERDAY_FEATURES)
    assert top[0]["feature"] in ("after_hours_frac", USERDAY_FEATURES[0])
    assert all(t["direction"] in ("raises", "lowers") for t in top)
    assert 0.0 <= day_risk(tiny_art, x) <= 1.0
    assert day_risk(tiny_art, None) == 0.0 and top_features(tiny_art, None) == []


def test_trails_sign_chain_and_cover_tracks_breaks_only_the_ledger():
    trails = Trails(Signer.generate(), backend="sim")
    ev = pd.Series({"ts": pd.Timestamp("2011-02-14 09:00:00"), "action": "s3:GetObject",
                    "resource": "arn:aws:s3:::sentinel-x/removable/PC-1"})
    x = np.zeros(34, np.float32)
    for i in range(12):
        verdict = "DENY" if i % 3 == 0 else "ALLOW"
        trails.record("HBO0413", ev, 0.9 if verdict == "DENY" else 0.1, 0.9, verdict, x)
    trails.flush()
    assert trails.verify("HBO0413")["intact"] and trails.committed == 12
    assert all(p["verdict"] == "ALLOW" for p in trails.plain)      # plain IAM logs ALLOW
    actions = trails.cover_tracks("HBO0413", n_delete=2, n_modify=2)
    assert len(actions) == 4
    v = trails.verify("HBO0413")
    assert not v["intact"] and v["firstDiscontinuity"] >= 0
    # The plain log lost the deleted records silently and shows only ALLOW.
    assert len(trails.plain) == 10 and all(p["verdict"] == "ALLOW" for p in trails.plain)
    # The first break is at or before the earliest tampered seq.
    assert v["firstDiscontinuity"] <= min(a["seq"] for a in actions) - 1 + 1


@pytest.mark.skipif(not (HAVE_SCENARIO and HAVE_MODELS), reason="needs demo/ and models/")
def test_scenario_replay_end_to_end():
    art = Artefacts.load()
    sc = Scenario.load()
    assert len(sc.days) > 5 and sc.events["label"].sum() > 0
    day = sc.days[len(sc.days) // 2]
    ev = sc.events[sc.events["day"] == day]
    x_day = sc.x_day(day)
    assert x_day is not None and x_day.shape == (len(USERDAY_FEATURES),)
    from sentinel.features.builder import FEATURE_NAMES

    x = ev[list(FEATURE_NAMES)].to_numpy(np.float32)
    scores = request_scores(art, x, ev["type"].to_numpy(), day_risk(art, x_day),
                            ev["resource_sensitivity"].to_numpy(np.float32))
    assert len(scores) == len(ev)
    assert set(scores["verdict"]) <= {"ALLOW", "ALLOW_OBSERVE", "STEPUP", "DENY"}
    assert (scores["r"] >= scores["r_request"]).all()            # propagation is a max


def _shim_up() -> bool:
    try:
        from sentinel.ledger.client import FabricLedger

        FabricLedger(timeout=3).health()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _shim_up(), reason="needs the Fabric test-network and shim")
def test_trails_on_the_live_fabric_ledger_detects_cover_tracks():
    trails = Trails(backend="fabric")
    assert trails.backend == "fabric"
    ev = pd.Series({"ts": pd.Timestamp("2011-02-14 09:00:00"), "action": "s3:GetObject",
                    "resource": "arn:aws:s3:::sentinel-x/removable/PC-1"})
    x = np.zeros(34, np.float32)
    for i in range(8):
        trails.record("DEMOTEST", ev, 0.9, 0.9, "DENY" if i % 2 else "ALLOW", x)
    trails.flush(timeout=120)
    assert trails.committed == 8 and trails.rejected == 0, trails.last_error
    assert trails.verify("DEMOTEST")["intact"]
    actions = trails.cover_tracks("DEMOTEST", n_delete=1, n_modify=1)
    assert len(actions) == 2
    v = trails.verify("DEMOTEST")
    assert not v["intact"] and v["firstDiscontinuity"] >= 0
