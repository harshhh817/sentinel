"""Sentinel demo dashboard: Plain IAM vs Sentinel, one CERT insider replayed day by day.

    make demo        # streamlit run sentinel/demo/app.py

Top to bottom: headline tiles, the risk timeline, the two systems side by side, the
evidence (ledger and cover-tracks), and the day-by-day table. Buttons act through
callbacks, so every number on the page is rendered after the action it reflects.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sentinel.demo.charts import (  # noqa: E402
    BANDS,
    HELD_BACK,
    VERDICT_LABEL,
    held_back_chart,
    risk_meter,
    risk_timeline,
    verdict_text,
)
from sentinel.demo.engine import (  # noqa: E402
    VERDICT_COLOUR,
    Artefacts,
    Scenario,
    Trails,
    day_risk,
    request_scores,
    top_features,
    ts_label,
)
from sentinel.features.builder import FEATURE_NAMES  # noqa: E402

st.set_page_config(page_title="Sentinel demo", layout="wide", page_icon="🛡️")

SOURCES = {"rf": "User-day RF — operating configuration",
           "max": "max(request hybrid r, day r) — paper's propagation"}
MIN_TAMPER = 6
CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stMetric"] {border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
  padding: 12px 14px; background: #ffffff;}
[data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #52514e;}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600;}
h1 {font-size: 1.7rem !important; margin-bottom: 0 !important;}
/* columns in the page body wrap instead of squeezing their contents into ellipses */
section.main [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
section.main [data-testid="column"] {min-width: 132px;}
</style>
"""


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
        ss.trails = Trails(backend=ss.get("backend", "auto"))
        ss.right_log = []          # per request: dict rows
        ss.day_rows = []           # per day summary
        ss.last_verify = None
        ss.last_top = []
        ss.last_R = 0.0
        ss.last_verdict = "ALLOW"
        ss.cover_actions = None


def current_source() -> str:
    return "max" if st.session_state.get("risk_source_choice") == SOURCES["max"] else "rf"


def settle(trails: Trails) -> None:
    """The reference ledger commits in microseconds: let the day land before it is shown.
    On Fabric each commit waits for its block, and the lag is part of what the demo shows."""
    if trails.backend == "sim":
        trails.flush(timeout=5.0)


def replay_day(art: Artefacts, sc: Scenario, day: str, source: str) -> None:
    ss = st.session_state
    ev = sc.events[sc.events["day"] == day]
    x_day = sc.x_day(day)
    r_day = day_risk(art, x_day)
    x = ev[list(FEATURE_NAMES)].to_numpy(np.float32)
    sens = ev["resource_sensitivity"].to_numpy(np.float32)
    scores = request_scores(art, x, ev["type"].to_numpy(), r_day, sens, source=source)
    for k, ((_, e), (_, s)) in enumerate(zip(ev.iterrows(), scores.iterrows(), strict=True)):
        rec = ss.trails.record(sc.principal, e, s["r"], s["R"], s["verdict"], x[k])
        ss.right_log.append({"ts": e["ts"], "action": e["action"], "resource": e["resource"],
                             "r": float(s["r"]), "R": float(s["R"]), "verdict": s["verdict"],
                             "seq": rec["seq"], "label": int(e["label"])})
    counts = scores["verdict"].value_counts().to_dict()
    peak = int(scores["R"].to_numpy().argmax()) if len(scores) else 0
    ss.last_R = float(scores["R"].iloc[peak]) if len(scores) else 0.0
    ss.last_verdict = scores["verdict"].iloc[peak] if len(scores) else "ALLOW"
    ss.last_top = top_features(art, x_day)
    ss.day_rows.append({"day": day, "events": len(ev), "malicious": int(ev["label"].sum()),
                        "day risk r": round(r_day, 3), "peak R": round(ss.last_R, 3),
                        "source": source, **{v: int(counts.get(v, 0)) for v in VERDICT_COLOUR}})
    settle(ss.trails)
    ss.last_verify = ss.trails.verify(sc.principal) if ss.trails.committed else None


# --- actions (callbacks run before the page is drawn) -----------------------------------


def on_step(n: int = 1) -> None:
    ss = st.session_state
    art, sc = artefacts(), scenario(None)
    for _ in range(n):
        if ss.day_idx >= len(sc.days):
            break
        replay_day(art, sc, sc.days[ss.day_idx], current_source())
        ss.day_idx += 1


def on_play() -> None:
    st.session_state.playing = not st.session_state.playing


def on_reset() -> None:
    st.session_state.principal = None


def on_cover() -> None:
    ss = st.session_state
    sc = scenario(None)
    settle(ss.trails)
    ss.cover_actions = ss.trails.cover_tracks(sc.principal)
    ss.last_verify = ss.trails.verify(sc.principal) if ss.trails.committed else None


# --- page sections ----------------------------------------------------------------------


def sidebar(sc: Scenario) -> float:
    ss, sb = st.session_state, st.sidebar
    sb.title("🛡️ Sentinel")
    sb.caption("Zero-trust access decisions with a tamper-evident audit trail")
    done = ss.day_idx >= len(sc.days)
    c1, c2 = sb.columns(2)
    c1.button("⏸ Pause" if ss.playing else "▶ Play", key="btn_play", on_click=on_play,
              disabled=done, use_container_width=True, type="primary")
    c2.button("Step day", key="btn_step", on_click=on_step, disabled=done or ss.playing,
              use_container_width=True)
    c3, c4 = sb.columns(2)
    c3.button("Skip 5 days", key="btn_skip", on_click=on_step, args=(5,),
              disabled=done or ss.playing, use_container_width=True)
    c4.button("Reset", key="btn_reset", on_click=on_reset, use_container_width=True)
    sb.progress(ss.day_idx / max(1, len(sc.days)),
                text=f"Day {ss.day_idx} of {len(sc.days)}"
                     + (f" · {sc.days[ss.day_idx - 1]}" if ss.day_idx else ""))
    speed = sb.slider("Replay speed (days per second)", 0.2, 5.0, 1.0, 0.2, key="speed")

    sb.divider()
    sb.radio("Risk source", list(SOURCES.values()), index=0, key="risk_source_choice")
    if current_source() == "rf":
        sb.caption("Random forest on the day's 75-feature vector; every request that day "
                   "carries the day's risk. Explained with SHAP.")
    else:
        sb.caption("The paper's propagation, r′ = max(request r, day r). The request-level "
                   "hybrid is a benign quantile with a 72 % false-positive rate at R ≥ 0.85, "
                   "so nearly every request is denied whatever the forest says — which is why "
                   "it is not the operating configuration.")

    sb.divider()
    sb.markdown(f"**Scenario** · CERT insider `{sc.principal}`")
    sb.caption(f"{len(sc.events):,} requests over {len(sc.days)} days, "
               f"{int(sc.events['label'].sum())} scripted as malicious")
    if sc.synthetic:
        sb.warning("Synthetic scenario and models (fresh-clone bootstrap). Run "
                   "`make demo-scenario` and `make train` on the CERT splits for the real one.")
    tr = ss.trails
    name = "Hyperledger Fabric" if tr.backend == "fabric" else "reference ledger (sim)"
    sb.markdown(f"**Ledger** · {name}")
    sb.caption(f"{tr.committed:,} committed · {tr.pending:,} pending · {tr.rejected:,} rejected")
    if tr.backend == "fabric":
        sb.caption("Each commit waits for its block, so the ledger lags the replay: "
                   "commitment is off the decision path, as in the paper.")
    if tr.last_error:
        sb.error(tr.last_error)
    return float(speed)


def headline(sc: Scenario, log: pd.DataFrame) -> None:
    ss = st.session_state
    held = int(log["verdict"].isin(HELD_BACK).sum()) if len(log) else 0
    malicious = int(log["label"].sum()) if len(log) else 0
    v = ss.last_verify
    if v is None:
        chain = "—"
    elif v["intact"]:
        chain = "✅ Intact"
    else:
        chain = "❌ Broken"
    tiles = [*st.columns(3), *st.columns(3)]
    tiles[0].metric("Day", f"{ss.day_idx} / {len(sc.days)}")
    tiles[1].metric("Requests", f"{len(log):,}", help="Requests replayed so far")
    tiles[2].metric("Malicious", f"{malicious:,}",
                    help="Requests the CERT scenario scripts as malicious, so far")
    tiles[3].metric("IAM allowed", f"{len(log):,}",
                    help="Plain IAM allows every request by an entitled principal")
    tiles[4].metric("Held back", f"{held:,}",
                    help="Requests Sentinel observed, stepped up or denied")
    where = "" if v is None or v["intact"] else (
        f" First discontinuity at seq {v['firstDiscontinuity'] + 1:,}.")
    tiles[5].metric("Audit chain", chain,
                    help="VerifyChain over this user's committed ledger records." + where)


def request_table(df: pd.DataFrame, cols: list[str]) -> None:
    view = df.tail(12).iloc[::-1].copy()
    view["ts"] = view["ts"].map(ts_label)
    view["verdict"] = view["verdict"].map(verdict_text)
    view["resource"] = view["resource"].str.replace("arn:aws:s3:::", "", regex=False)
    st.dataframe(
        view[cols], use_container_width=True, hide_index=True, height=38 + 35 * 12,
        column_config={
            "ts": st.column_config.TextColumn("Time", width="medium"),
            "action": st.column_config.TextColumn("Action"),
            "resource": st.column_config.TextColumn("Resource", width="medium"),
            "r": st.column_config.NumberColumn("Risk r", format="%.3f"),
            "R": st.column_config.NumberColumn("Effective R", format="%.3f"),
            "verdict": st.column_config.TextColumn("Verdict"),
            "seq": st.column_config.NumberColumn("Seq", format="%d"),
        })


def systems(log: pd.DataFrame) -> None:
    ss = st.session_state
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.subheader("Plain IAM")
        st.caption("Entitlement only. Every request by an entitled principal is allowed, and "
                   "the log is a file an administrator can edit.")
        st.metric("Verdict on every request", verdict_text("ALLOW"))
        if len(log):
            st.markdown("**Latest requests**")
            request_table(log.assign(verdict="ALLOW"), ["ts", "action", "resource", "verdict"])
    with right, st.container(border=True):
        st.subheader("Sentinel")
        st.caption("Each request is scored against the user's own baseline; the decision and "
                   "its evidence are signed, chained and committed to a ledger.")
        st.metric("Peak verdict today", verdict_text(ss.last_verdict))
        st.plotly_chart(risk_meter(ss.last_R, ss.last_verdict), use_container_width=True,
                        config={"displayModeBar": False}, key="meter")
        if len(log):
            counts = log["verdict"].value_counts()
            for col, v in zip(st.columns(4), VERDICT_COLOUR, strict=True):
                col.metric(verdict_text(v), f"{int(counts.get(v, 0)):,}")
            if ss.last_top:
                st.markdown("**What raised today's risk** — top 3 by SHAP on the user-day "
                            "forest")
                for f in ss.last_top:
                    st.markdown(f"- `{f['feature']}` {f['direction']} risk "
                                f"(SHAP {f['shap']:+.3f}, value {f['value']:.2f})")
            st.markdown("**Latest requests**")
            request_table(log, ["ts", "action", "r", "R", "verdict", "seq"])


def evidence(sc: Scenario) -> None:
    ss = st.session_state
    st.subheader("Evidence")
    tr, v = ss.trails, ss.last_verify
    a, b = st.columns([1, 2], gap="large")
    with a:
        st.caption("Play the privileged administrator who wants this to go away: delete three "
                   "suspicious records and rewrite three verdicts to Allow, in both stores.")
        ready = tr.committed >= MIN_TAMPER and not ss.cover_actions
        st.button("🕵️ Cover tracks", key="btn_cover", type="primary", on_click=on_cover,
                  disabled=not ready, use_container_width=True)
        if tr.committed < MIN_TAMPER:
            st.caption(f"Needs at least {MIN_TAMPER} committed records "
                       f"({tr.committed} so far).")
        elif ss.cover_actions:
            st.caption("Already tampered with. Reset to run it again.")
    with b:
        if v is None:
            st.info("Nothing committed yet. Step a day to write the first records.")
        elif not ss.cover_actions:
            st.success(f"Ledger holds {v['records']:,} records for this user. VerifyChain "
                       f"walked the chain from genesis: intact.", icon="✅")
        else:
            acts = ss.cover_actions
            n_del = sum(x["op"] == "delete" for x in acts)
            n_mod = sum(x["op"] == "modify" for x in acts)
            st.error(f"The administrator deleted {n_del} records and rewrote {n_mod} verdicts "
                     f"to Allow, in both stores.", icon="🕵️")
            p, q = st.columns(2)
            p.markdown(f"**Plain IAM log**  \n{len(tr.plain):,} records, nothing to check them "
                       f"against. The edits are invisible.")
            if v["intact"]:
                q.markdown("**Ledger**  \nVerifyChain: intact.")
            else:
                q.markdown(f"**Ledger**  \nVerifyChain: first discontinuity at seq "
                           f"{v['firstDiscontinuity'] + 1:,} — {v['reason']}.")
                recs = pd.DataFrame(tr.ledger_records(sc.principal))
                bad = v["firstDiscontinuity"]
                if len(recs):
                    view = recs[["seq", "ts", "action", "verdict", "prevHash"]].copy()
                    view["prevHash"] = view["prevHash"].str[:12] + "…"
                    view["verdict"] = view["verdict"].map(verdict_text)
                    lo, hi = max(0, bad - 3), min(len(view), bad + 4)
                    view = view.iloc[lo:hi]
                    view["note"] = ["⟵ chain breaks here" if i == bad else ""
                                    for i in view.index]
                    st.dataframe(
                        view, use_container_width=True, hide_index=True,
                        column_config={
                            "seq": st.column_config.NumberColumn("Seq", format="%d"),
                            "ts": st.column_config.TextColumn("Time"),
                            "action": st.column_config.TextColumn("Action"),
                            "verdict": st.column_config.TextColumn("Verdict"),
                            "prevHash": st.column_config.TextColumn("Previous hash"),
                            "note": st.column_config.TextColumn("", width="medium")})


def day_table() -> None:
    rows = pd.DataFrame(st.session_state.day_rows)
    st.subheader("Day by day")
    if not len(rows):
        st.caption("One row per replayed day will appear here.")
        return
    view = rows.rename(columns={v: VERDICT_LABEL[v] for v in VERDICT_COLOUR}).iloc[::-1]
    st.dataframe(
        view, use_container_width=True, hide_index=True,
        column_config={
            "day": st.column_config.TextColumn("Day"),
            "events": st.column_config.NumberColumn("Requests", format="%d"),
            "malicious": st.column_config.NumberColumn("Scripted-malicious", format="%d"),
            "day risk r": st.column_config.NumberColumn("Day risk r (forest)", format="%.3f"),
            "peak R": st.column_config.ProgressColumn("Peak effective R", format="%.2f",
                                                       min_value=0.0, max_value=1.0),
            "source": st.column_config.TextColumn("Risk source"),
        })


def guide() -> None:
    with st.expander("How to read this page"):
        bands = pd.DataFrame(
            [{"Effective risk R": f"{lo:.2f} – {hi:.2f}", "Verdict": verdict_text(v),
              "What happens": w}
             for (v, lo, hi), w in zip(BANDS, (
                 "Credential for 60 minutes, scope as requested",
                 "Allowed and observed: credential for 30 minutes, verbose logging",
                 "Signed challenge; 15 minutes, read-only where possible",
                 "No credential issued"), strict=True)])
        st.dataframe(bands, use_container_width=True, hide_index=True)
        st.markdown(
            "- **Risk r** is the model's score for the request. **Effective R** is what the "
            "trust algorithm makes of it once the resource's sensitivity is taken into "
            "account; the verdict comes from R.\n"
            "- **Plain IAM** checks entitlement only, so it allows every request this user "
            "makes.\n"
            "- Every Sentinel decision becomes a record with a per-user sequence number, the "
            "hash of the previous record, a salted digest of the features and a P-256 "
            "signature. **VerifyChain** re-walks that chain from the first record.")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    art = artefacts()
    sc = scenario(None)
    init_state(sc)
    ss = st.session_state
    speed = sidebar(sc)

    st.title("Plain IAM vs Sentinel")
    st.caption(f"The same user, the same {len(sc.days)} days, two systems. "
               f"Replaying CERT r4.2 insider {sc.principal} as CloudTrail-style requests.")
    log = pd.DataFrame(ss.right_log)
    headline(sc, log)
    if not ss.day_idx:
        st.info("Press **Step day** or **Play** in the sidebar to start the replay.", icon="▶️")

    rows = pd.DataFrame(ss.day_rows)
    cfg = {"displayModeBar": False}
    st.plotly_chart(risk_timeline(rows, sc.days), use_container_width=True, config=cfg,
                    key="timeline")
    st.plotly_chart(held_back_chart(rows, sc.days), use_container_width=True, config=cfg,
                    key="held_back")

    systems(log)
    st.divider()
    evidence(sc)
    st.divider()
    day_table()
    guide()

    if ss.playing and ss.day_idx < len(sc.days):
        time.sleep(1.0 / speed)
        replay_day(art, sc, sc.days[ss.day_idx], current_source())
        ss.day_idx += 1
        st.rerun()
    elif ss.playing:
        ss.playing = False
        st.rerun()


main()
