"""Chart builders for the dashboard. Pure functions: data in, plotly Figure out.

Verdicts are states, so they wear a fixed status palette and are never colour-only: every
chart carries a legend and hover, every table cell an icon and a label, and the
day-by-day table is the table-view twin of both charts.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

STATUS = {"ALLOW": "#0ca30c", "ALLOW_OBSERVE": "#fab219", "STEPUP": "#ec835a",
          "DENY": "#d03b3b"}
VERDICT_LABEL = {"ALLOW": "Allow", "ALLOW_OBSERVE": "Observe", "STEPUP": "Step-up",
                 "DENY": "Deny"}
VERDICT_ICON = {"ALLOW": "🟢", "ALLOW_OBSERVE": "🟡", "STEPUP": "🟠", "DENY": "🔴"}
# Table III: lower bound of each band on the effective risk R.
BANDS = (("ALLOW", 0.0, 0.40), ("ALLOW_OBSERVE", 0.40, 0.65), ("STEPUP", 0.65, 0.85),
         ("DENY", 0.85, 1.0))
HELD_BACK = ("ALLOW_OBSERVE", "STEPUP", "DENY")

SERIES = "#2a78d6"        # categorical slot 1
ACCENT = "#eb6834"        # categorical slot 2: the days the scenario scripts as malicious
SURFACE = "#fcfcfb"
MUTED = "#898781"
GRID = "rgba(137,135,129,0.22)"
AXIS = "rgba(137,135,129,0.55)"
DAY_MS = 86_400_000       # bar width on a date axis is in milliseconds
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def verdict_text(verdict: str) -> str:
    return f"{VERDICT_ICON.get(verdict, '')} {VERDICT_LABEL.get(verdict, verdict)}".strip()


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _frame(fig: go.Figure, title: str, height: int, days: list[str]) -> go.Figure:
    fig.update_layout(
        height=height, margin={"l": 8, "r": 8, "t": 40, "b": 8},
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        font={"family": FONT}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        # below the plot: a narrow chart wraps a top legend into its own title
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0, "xanchor": "left"},
        hovermode="x unified", showlegend=True)
    if days:
        lo = pd.Timestamp(days[0]) - pd.Timedelta(days=1)
        hi = pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
        fig.update_xaxes(range=[lo, hi])
    fig.update_xaxes(type="date", showgrid=False, linecolor=AXIS, tickfont={"color": MUTED},
                     tickformat="%d %b")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont={"color": MUTED})
    return fig


def risk_timeline(day_rows: pd.DataFrame, days: list[str]) -> go.Figure:
    """Peak effective risk R per replayed day, against the Table III band boundaries."""
    fig = go.Figure()
    for verdict, lo, _ in BANDS[1:]:
        fig.add_hline(y=lo, line_width=1, line_color=AXIS,
                      annotation_text=f"{VERDICT_LABEL[verdict].lower()} ≥ {lo:.2f}",
                      annotation_position="top left",
                      annotation_font={"size": 11, "color": MUTED})
    if len(day_rows):
        d = day_rows.assign(x=pd.to_datetime(day_rows["day"]))
        custom = d[["day risk r", "events", "malicious"]].to_numpy()
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["peak R"], mode="lines", name="Peak effective risk R",
            line={"color": SERIES, "width": 2, "shape": "linear"}, customdata=custom,
            hovertemplate=("R %{y:.2f} · day risk r %{customdata[0]:.3f} · "
                           "%{customdata[1]} requests<extra></extra>")))
        m = d[d["malicious"] > 0]
        fig.add_trace(go.Scatter(
            x=m["x"], y=m["peak R"], mode="markers",
            name="Day with scripted-malicious events",
            marker={"color": ACCENT, "size": 9, "line": {"color": SURFACE, "width": 2}},
            customdata=m[["malicious"]].to_numpy(),
            hovertemplate="%{customdata[0]} scripted-malicious events<extra></extra>"))
    fig.update_yaxes(range=[0, 1.04], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return _frame(fig, "Effective risk per day", 320, days)


def held_back_chart(day_rows: pd.DataFrame, days: list[str]) -> go.Figure:
    """Requests per day that Sentinel did not plainly allow. Quiet days stay empty."""
    fig = go.Figure()
    for verdict in HELD_BACK:
        y = day_rows[verdict] if len(day_rows) else []
        x = pd.to_datetime(day_rows["day"]) if len(day_rows) else []
        fig.add_trace(go.Bar(
            x=x, y=y, name=VERDICT_LABEL[verdict], width=DAY_MS * 0.9,
            marker={"color": STATUS[verdict], "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{y} " + VERDICT_LABEL[verdict].lower() + "<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.35, barcornerradius=4)
    fig.update_yaxes(rangemode="tozero", title={"text": "requests", "font": {"color": MUTED}})
    return _frame(fig, "Requests held back per day", 250, days)


def risk_meter(R: float, verdict: str) -> go.Figure:
    """The current day's peak R as a bullet against the four bands."""
    fig = go.Figure(go.Indicator(
        mode="number+gauge", value=float(R),
        number={"valueformat": ".2f", "font": {"size": 40}},
        gauge={"shape": "bullet", "borderwidth": 0,
               "axis": {"range": [0, 1], "tickvals": [0, 0.40, 0.65, 0.85, 1],
                        "tickfont": {"color": MUTED}},
               "bar": {"color": STATUS[verdict], "thickness": 0.42},
               "steps": [{"range": [lo, hi], "color": _rgba(STATUS[v], 0.18)}
                         for v, lo, hi in BANDS]}))
    fig.update_layout(height=110, margin={"l": 8, "r": 8, "t": 8, "b": 28},
                      font={"family": FONT}, paper_bgcolor="rgba(0,0,0,0)")
    return fig
