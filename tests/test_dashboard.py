"""The dashboard itself, driven headless through Streamlit's AppTest: every control."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.demo.engine import DEMO_DIR

pytestmark = pytest.mark.skipif(
    not ((DEMO_DIR / "scenario_meta.json").exists()
         and (Path("models") / "engine_seed0.json").exists()),
    reason="needs demo/ and models/ (make demo-bootstrap)")

APP = str(Path(__file__).resolve().parent.parent / "sentinel" / "demo" / "app.py")


@pytest.fixture
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["backend"] = "sim"
    return at.run()


def tile(at, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_loads_clean_with_nothing_replayed(app):
    assert not app.exception
    assert tile(app, "Requests") == "0" and tile(app, "Audit chain") == "—"
    assert tile(app, "Day").startswith("0 / ")
    assert app.button(key="btn_cover").disabled
    assert any("Press **Step day**" in i.value for i in app.info)


def test_step_day_replays_one_day_and_the_chain_is_intact(app):
    app.button(key="btn_step").click().run()
    assert not app.exception
    assert tile(app, "Day").startswith("1 / ")
    n = int(tile(app, "Requests").replace(",", ""))
    assert n > 0 and tile(app, "IAM allowed") == tile(app, "Requests")
    assert tile(app, "Audit chain") == "✅ Intact"
    # the sidebar ledger counters reflect the day that was just replayed
    assert any(f"{n:,} committed · 0 pending · 0 rejected" in c.value
               for c in app.sidebar.caption)
    assert len(app.session_state.day_rows) == 1


def test_skip_five_days_and_reset(app):
    app.button(key="btn_skip").click().run()
    assert tile(app, "Day").startswith("5 / ") and len(app.session_state.day_rows) == 5
    app.button(key="btn_reset").click().run()
    assert not app.exception
    assert tile(app, "Day").startswith("0 / ") and tile(app, "Requests") == "0"
    assert app.session_state.day_rows == [] and app.session_state.cover_actions is None


def test_operating_configuration_allows_a_quiet_day_and_the_paper_rule_does_not(app):
    app.button(key="btn_step").click().run()
    first = app.session_state.day_rows[0]
    assert first["source"] == "rf"
    app.sidebar.radio(key="risk_source_choice").set_value(
        "max(request hybrid r, day r) — paper's propagation").run()
    app.button(key="btn_step").click().run()
    second = app.session_state.day_rows[1]
    assert not app.exception and second["source"] == "max"
    held = second["ALLOW_OBSERVE"] + second["STEPUP"] + second["DENY"]
    assert held > 0.5 * second["events"]              # the request-level artefact
    assert first["ALLOW"] >= 0.5 * first["events"] or first["malicious"] > 0


def test_cover_tracks_breaks_the_ledger_chain_and_only_the_ledger(app):
    app.button(key="btn_skip").click().run()
    before = int(tile(app, "Requests").replace(",", ""))
    assert not app.button(key="btn_cover").disabled
    app.button(key="btn_cover").click().run()
    assert not app.exception
    assert tile(app, "Audit chain") == "❌ Broken"
    assert any("first discontinuity at seq" in m.value for m in app.markdown)
    assert any("deleted" in e.value and "rewrote" in e.value for e in app.error)
    assert len(app.session_state.trails.plain) < before
    assert all(p["verdict"] == "ALLOW" for p in app.session_state.trails.plain)
    assert app.button(key="btn_cover").disabled          # one tamper per session


def test_play_runs_to_the_end_and_stops(app):
    app.session_state["day_idx"] = len(app.session_state.day_rows)
    app.sidebar.slider(key="speed").set_value(5.0).run()
    app.button(key="btn_skip").click().run()
    app.button(key="btn_play").click().run()
    assert not app.exception
    assert app.session_state.day_idx >= 6


def test_chain_status_follows_the_ledger_not_the_moment_of_replay(app):
    """On Fabric the ledger commits after the day is drawn. Simulate that: a stale status
    from before anything was committed must be replaced on the next render."""
    app.button(key="btn_step").click().run()
    app.session_state["last_verify"] = None
    app.session_state["verified_at"] = -1
    app.run()
    assert not app.exception and tile(app, "Audit chain") == "✅ Intact"
    assert app.session_state.verified_at == app.session_state.trails.committed
    assert any("VerifyChain walked the chain" in s.value for s in app.success)
