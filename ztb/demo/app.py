"""ZTBAudit demo: Plain IAM vs ZTBAudit, one CERT insider replayed day by day.

    make demo        # streamlit run ztb/demo/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztb.demo.engine import (  # noqa: E402
    VERDICT_COLOUR,
    Artefacts,
    Scenario,
    Trails,
    day_risk,
    request_scores,
    top_features,
)
from ztb.features.builder import FEATURE_NAMES  # noqa: E402

st.set_page_config(page_title="ZTBAudit demo", layout="wide", page_icon="🛡️")


@st.cache_resource(show_spinner="loading models ...")
def artefacts() -> Artefacts:
    return Artefacts.load()


@st.cache_resource(show_spinner="loading scenario ...")
def scenario(principal: str | None) -> Scenario:
    return Scenario.load(principal=principal)


def init_state(sc: Scenario) -> None:
    ss = st.session_state
    if ss.get("principal") != sc.principal:
        ss.principal = sc.principal
        ss.day_idx = 0
        ss.playing = False
        ss.trails = Trails(backend=st.session_state.get("backend", "auto"))
        ss.right_log = []          # per request: dict rows
        ss.day_rows = []           # per day summary
        ss.last_verify = None
        ss.last_top = []
        ss.last_R = 0.0
        ss.last_verdict = "ALLOW"


def replay_day(art: Artefacts, sc: Scenario, day: str) -> None:
    ss = st.session_state
    ev = sc.events[sc.events["day"] == day]
    x_day = sc.x_day(day)
    r_day = day_risk(art, x_day)
    top = top_features(art, x_day)
    x = ev[list(FEATURE_NAMES)].to_numpy(np.float32)
    sens = ev["resource_sensitivity"].to_numpy(np.float32)
    scores = request_scores(art, x, ev["type"].to_numpy(), r_day, sens)
    for (i, e), (_, s) in zip(ev.iterrows(), scores.iterrows(), strict=True):
        rec = ss.trails.record(sc.principal, e, s["r"], s["R"], s["verdict"],
                               x[ev.index.get_loc(i)])
        ss.right_log.append({"ts": e["ts"], "action": e["action"], "type": e["type"],
                             "r": round(float(s["r"]), 3), "R": round(float(s["R"]), 3),
                             "verdict": s["verdict"], "seq": rec["seq"], "recId": rec["recId"],
                             "label": int(e["label"])})
    counts = scores["verdict"].value_counts().to_dict()
    ss.day_rows.append({"day": day, "events": len(ev), "malicious": int(ev["label"].sum()),
                        "day risk (RF)": round(r_day, 3), **{k: counts.get(k, 0) for k in
                                                                VERDICT_COLOUR}})
    ss.last_top, ss.last_R = top, float(scores["R"].max()) if len(scores) else 0.0
    ss.last_verdict = scores["verdict"].iloc[int(scores["R"].idxmax())] if len(scores) else "ALLOW"
    ss.last_verify = ss.trails.verify(sc.principal) if ss.trails.committed else None


def gauge(R: float, verdict: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=R, number={"valueformat": ".2f"},
        title={"text": f"effective risk R — {verdict}"},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": VERDICT_COLOUR[verdict]},
               "steps": [{"range": [0, 0.40], "color": "#e8f5e9"},
                         {"range": [0.40, 0.65], "color": "#fff8e1"},
                         {"range": [0.65, 0.85], "color": "#fff3e0"},
                         {"range": [0.85, 1.0], "color": "#ffebee"}]}))
    fig.update_layout(height=230, margin={"l": 20, "r": 20, "t": 50, "b": 10})
    return fig


def main() -> None:
    art = artefacts()
    st.sidebar.title("ZTBAudit demo")
    meta_principal = None
    sc = scenario(meta_principal)
    init_state(sc)
    ss = st.session_state
    speed = st.sidebar.slider("replay speed (days / second)", 0.2, 5.0, 1.0, 0.2)
    st.sidebar.caption(f"scenario: **{sc.principal}** · {len(sc.events):,} events over "
                       f"{len(sc.days)} days · {int(sc.events['label'].sum())} scripted-malicious")
    st.sidebar.caption("operating configuration: random forest on user-day vectors "
                       "(SHAP top-3), propagated to each request as r′ = max(request r, day r)")
    tr = ss.trails
    st.sidebar.markdown(f"**ledger: {tr.backend}** · committed {tr.committed} · pending "
                       f"{tr.pending} · rejected {tr.rejected}")
    if tr.backend == "fabric":
        st.sidebar.caption("Hyperledger Fabric test-network via the gateway shim; each commit "
                           "waits for its block, so the ledger lags the replay (asynchronous "
                           "committer, as in the paper).")
    if tr.last_error:
        st.sidebar.error(tr.last_error)
    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("▶ play" if not ss.playing else "⏸ pause"):
        ss.playing = not ss.playing
    if c2.button("step day"):
        if ss.day_idx < len(sc.days):
            replay_day(art, sc, sc.days[ss.day_idx])
            ss.day_idx += 1
    if c3.button("reset"):
        ss.principal = None
        st.rerun()
    st.sidebar.progress(ss.day_idx / max(1, len(sc.days)),
                        text=f"day {ss.day_idx}/{len(sc.days)}")

    left, right = st.columns(2)
    with left:
        st.subheader("Plain IAM")
        st.caption("entitlement only: every request by an entitled principal is ALLOW; the log "
                   "is a mutable file")
        plain = pd.DataFrame(ss.trails.plain)
        if len(plain):
            st.metric("records in log", len(plain))
            st.dataframe(plain[["ts", "action", "resource", "verdict"]].tail(12),
                         use_container_width=True, hide_index=True)
    with right:
        st.subheader("ZTBAudit")
        st.plotly_chart(gauge(ss.last_R, ss.last_verdict), use_container_width=True)
        if ss.last_top:
            st.markdown("**top-3 contributing features (SHAP, user-day RF)**")
            for f in ss.last_top:
                st.markdown(f"- `{f['feature']}` {f['direction']} risk "
                            f"(SHAP {f['shap']:+.3f}, value {f['value']:.2f})")
        log = pd.DataFrame(ss.right_log)
        if len(log):
            counts = log["verdict"].value_counts()
            cols = st.columns(4)
            for c, v in zip(cols, VERDICT_COLOUR, strict=True):
                c.metric(v, int(counts.get(v, 0)))
            show = log[["ts", "action", "type", "r", "R", "verdict", "seq"]].tail(12)
            st.dataframe(show.style.map(lambda v: f"color:{VERDICT_COLOUR.get(v, '')}",
                                        subset=["verdict"]),
                         use_container_width=True, hide_index=True)
        if ss.last_verify:
            v = ss.last_verify
            status = ("✅ intact" if v["intact"]
                      else f"❌ broken at seq {v['firstDiscontinuity'] + 1}")
            st.markdown(f"ledger ({ss.trails.backend}): **{v['records']} committed records**, "
                        f"VerifyChain → {status}")

    if ss.day_rows:
        st.markdown("#### day-by-day")
        st.dataframe(pd.DataFrame(ss.day_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    b1, b2 = st.columns([1, 3])
    can_tamper = ss.trails.committed >= 6
    b1.caption("" if can_tamper else f"needs ≥ 6 committed records ({ss.trails.committed})")
    if b1.button("🕵️ Cover tracks", type="primary", disabled=not can_tamper):
        actions = ss.trails.cover_tracks(sc.principal)
        ss.last_verify = ss.trails.verify(sc.principal) if ss.trails.committed else None
        ss.cover_actions = actions
    if ss.get("cover_actions"):
        acts = ss.cover_actions
        v = ss.last_verify
        n_del = sum(a["op"] == "delete" for a in acts)
        n_mod = sum(a["op"] == "modify" for a in acts)
        verdict_line = ("intact" if v["intact"] else
                        f"❌ first discontinuity at seq {v['firstDiscontinuity'] + 1} "
                        f"— {v['reason']}")
        with b2:
            st.error(f"A3 deleted {n_del} records and rewrote {n_mod} verdicts to ALLOW in "
                     f"**both** stores.")
            st.markdown(f"**Plain IAM log:** {len(ss.trails.plain)} records, nothing to check "
                        f"against — the edits are invisible.")
            st.markdown(f"**Ledger VerifyChain:** {verdict_line}")
            recs = pd.DataFrame(ss.trails.ledger_records(sc.principal))
            if len(recs) and not v["intact"]:
                bad = v["firstDiscontinuity"]
                view = recs[["seq", "ts", "action", "verdict", "prevHash"]].copy()
                view["prevHash"] = view["prevHash"].str[:12] + "…"
                lo, hi = max(0, bad - 3), min(len(view), bad + 4)
                st.dataframe(view.iloc[lo:hi].style.apply(
                    lambda row: ["background-color:#ffcdd2" if row.name == bad else ""] * len(row),
                    axis=1), use_container_width=True, hide_index=True)

    if ss.playing and ss.day_idx < len(sc.days):
        replay_day(art, sc, sc.days[ss.day_idx])
        ss.day_idx += 1
        time.sleep(1.0 / speed)
        st.rerun()
    elif ss.playing:
        ss.playing = False


main()
