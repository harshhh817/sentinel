"""Dashboard chart builders: pure functions, so they are tested without Streamlit."""

from __future__ import annotations

import pandas as pd

from sentinel.demo.charts import (
    BANDS,
    HELD_BACK,
    STATUS,
    held_back_chart,
    risk_meter,
    risk_timeline,
    verdict_text,
)
from sentinel.demo.engine import VERDICT_COLOUR
from sentinel.risk.trust import band_lookup

DAYS = ["2011-02-11", "2011-02-14", "2011-02-15"]
ROWS = pd.DataFrame([
    {"day": "2011-02-11", "events": 97, "malicious": 0, "day risk r": 0.005, "peak R": 0.01,
     "ALLOW": 97, "ALLOW_OBSERVE": 0, "STEPUP": 0, "DENY": 0},
    {"day": "2011-02-14", "events": 113, "malicious": 9, "day risk r": 0.43, "peak R": 0.75,
     "ALLOW": 2, "ALLOW_OBSERVE": 0, "STEPUP": 111, "DENY": 0},
])


def test_bands_match_the_trust_algorithm():
    for verdict, lo, hi in BANDS:
        assert band_lookup(lo).verdict == verdict
        assert band_lookup((lo + hi) / 2).verdict == verdict
    assert BANDS[0][1] == 0.0 and BANDS[-1][2] == 1.0
    assert [hi for _, _, hi in BANDS[:-1]] == [lo for _, lo, _ in BANDS[1:]]


def test_one_palette_for_verdicts_everywhere():
    assert STATUS == VERDICT_COLOUR
    assert set(HELD_BACK) == set(STATUS) - {"ALLOW"}
    assert verdict_text("STEPUP") == "🟠 Step-up" and verdict_text("ALLOW") == "🟢 Allow"


def test_timeline_has_one_axis_a_legend_and_marks_the_malicious_days():
    fig = risk_timeline(ROWS, DAYS)
    assert [t.name for t in fig.data] == ["Peak effective risk R",
                                          "Day with scripted-malicious events"]
    assert list(fig.data[0].y) == [0.01, 0.75] and fig.data[0].line.width == 2
    assert list(fig.data[1].y) == [0.75] and fig.data[1].marker.size >= 8
    assert fig.layout.showlegend and tuple(fig.layout.yaxis.range) == (0, 1.04)
    assert "yaxis2" not in fig.layout.to_plotly_json()          # one axis, never two
    assert len(fig.layout.shapes) == 3                     # the three band boundaries


def test_held_back_chart_stacks_only_what_was_not_plainly_allowed():
    fig = held_back_chart(ROWS, DAYS)
    assert [t.name for t in fig.data] == ["Observe", "Step-up", "Deny"]
    assert list(fig.data[1].y) == [0, 111] and fig.layout.barmode == "stack"
    assert [t.marker.color for t in fig.data] == [STATUS[v] for v in HELD_BACK]


def test_charts_render_before_the_first_day_is_replayed():
    empty = pd.DataFrame()
    assert len(risk_timeline(empty, DAYS).data) == 0
    assert len(held_back_chart(empty, DAYS).data) == 3
    assert risk_meter(0.0, "ALLOW").data[0].value == 0.0
    assert risk_meter(0.9, "DENY").data[0].gauge.bar.color == STATUS["DENY"]
