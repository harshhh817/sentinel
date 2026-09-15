"""Module 1 tests: mapping, labels, baseline decay, the 34-dim vector, and the split."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from sentinel.config import N_FEATURES
from sentinel.features.baseline import BaselineStore, PrincipalProfile, _decay_factor
from sentinel.features.builder import (
    CONTINUOUS_FEATURES,
    FEATURE_NAMES,
    Standardiser,
    build_vector,
    geodesic_km,
    host_identity,
    observe_event,
)
from sentinel.features.cert_mapper import stream_events
from sentinel.features.labels import load_labels
from sentinel.features.schema import CERT_HEADERS, CloudEvent, SchemaError, validate_header
from tests.fixtures import write_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_corpus(tmp_path_factory.mktemp("cert") / "r4.2")


# --- schema ----------------------------------------------------------------


def test_header_validation_rejects_wrong_schema():
    validate_header("logon", list(CERT_HEADERS["logon"]))
    with pytest.raises(SchemaError, match="header mismatch"):
        validate_header("logon", ["id", "date", "user", "pc"])


# --- mapping ---------------------------------------------------------------


def test_stream_is_ordered_by_timestamp(corpus):
    events = list(stream_events(corpus["root"]))
    assert events, "fixture produced no events"
    stamps = [e.ts for e in events]
    assert stamps == sorted(stamps), "k-way merge did not yield timestamp order"


def test_every_row_is_mapped_exactly_once(corpus):
    events = list(stream_events(corpus["root"]))
    assert len(events) == corpus["total_events"]
    assert len({e.event_id for e in events}) == corpus["total_events"]


def test_mapping_targets_match_the_paper(corpus):
    events = list(stream_events(corpus["root"]))
    actions = {e.source: {ev.action for ev in events if ev.source == e.source} for e in events}
    assert actions["logon"] <= {"sts:AssumeRole", "sts:SessionEnd"}
    assert actions["device"] == {"s3:GetObject"}
    assert actions["file"] == {"s3:GetObject"}     # r4.2: every file event is a copy to USB
    assert actions["http"] == {"execute-api:Invoke"}
    # Removable-media and external-proxy traffic carries the egress marker.
    assert all(e.egress for e in events if e.source in {"device", "file", "http"})
    assert all(e.bytes_read > 0 for e in events if e.source == "file")


def test_timestamps_and_identities_are_preserved(corpus):
    events = list(stream_events(corpus["root"]))
    assert {e.principal for e in events} == corpus["users"]
    assert min(e.ts for e in events) == corpus["start"]


def test_resources_carry_the_org_unit(corpus):
    events = [e for e in stream_events(corpus["root"]) if e.resource.startswith("arn:aws:s3")]
    assert events
    assert all("sentinel-" in e.resource for e in events)


def test_until_cutoff_limits_the_scan(corpus):
    cutoff = corpus["start"] + timedelta(days=5)
    events = list(stream_events(corpus["root"], until=cutoff))
    assert events
    assert all(e.ts <= cutoff for e in events)
    assert len(events) < corpus["total_events"]


# --- labels ----------------------------------------------------------------


def test_labels_are_read_from_nested_answers_tree(corpus):
    labels = load_labels(corpus["root"] / "answers")
    assert labels.event_ids == corpus["malicious_ids"]
    assert labels.insiders == {"CDE1846"}
    assert "r4.2-2" in labels.scenarios


def test_labels_ignore_other_releases_in_the_shared_tree(corpus):
    """answers/ on KiltHub is shared across r2..r6.2; only the requested release counts."""
    labels = load_labels(corpus["root"] / "answers", release="4.2")
    assert "ZZZ9999" not in labels.insiders          # r5.2 insider row filtered out
    assert "r4.1-1" not in labels.scenarios           # r4.1 scenario file skipped
    other = load_labels(corpus["root"] / "answers", release="4.1")
    assert other.insiders == set()
    assert "r4.1-1" in other.scenarios and len(other) == 1


def test_labels_attach_to_the_right_events(corpus):
    labels = load_labels(corpus["root"] / "answers")
    events = list(stream_events(corpus["root"], labels=set(labels.keys)))
    positives = [e for e in events if e.label == 1]
    assert {e.event_id for e in positives} == corpus["malicious_ids"]
    assert {e.principal for e in positives} == {"CDE1846"}


def test_missing_answers_directory_yields_no_labels(tmp_path):
    labels = load_labels(tmp_path / "nope")
    assert len(labels) == 0


# --- baseline --------------------------------------------------------------


def test_ewma_half_life_is_thirty_days():
    assert _decay_factor(30 * 86400) == pytest.approx(0.5)
    assert _decay_factor(60 * 86400) == pytest.approx(0.25)
    assert _decay_factor(0) == 1.0


def test_decayed_counts_age_toward_the_request():
    p = PrincipalProfile("u")
    t0 = datetime(2010, 1, 1)
    p.observe(t0, "s3:GetObject", "arn:r", "PC-1", "AS1", is_read=True)
    assert p.action_frequency("s3:GetObject", t0) == pytest.approx(1.0)
    # After 30 days the weight has halved, but the *frequency* is unchanged because
    # the numerator and denominator decay together.
    p._decay_to(t0 + timedelta(days=30))
    assert p.total == pytest.approx(0.5)
    assert p.action_frequency("s3:GetObject", t0 + timedelta(days=30)) == pytest.approx(1.0)


def test_sliding_windows_are_anchored_on_the_request():
    p = PrincipalProfile("u")
    t0 = datetime(2010, 1, 1, 12, 0, 0)
    for i in range(10):
        p.observe(t0 + timedelta(seconds=30 * i), "a", f"r{i}", "PC", "AS", is_read=True)
    last = t0 + timedelta(seconds=270)
    c1, c15, c60, resources, actions = p.window_counts(last)
    assert c1 == 3          # events within the trailing 60 s, not a wall-clock minute
    assert c15 == c60 == 10
    assert resources == 10
    assert actions == 1
    # An hour later the windows are empty again.
    assert p.window_counts(last + timedelta(hours=2))[2] == 0


def test_novelty_flags_track_first_occurrence():
    p = PrincipalProfile("u")
    t0 = datetime(2010, 1, 1)
    assert "a" not in p.seen_actions
    p.observe(t0, "a", "r", "PC", "AS", is_read=True)
    assert "a" in p.seen_actions and "r" in p.seen_resources and "AS" in p.seen_asns


# --- feature builder -------------------------------------------------------


def _event(ts, action="s3:GetObject", resource="arn:aws:s3:::sentinel-research/x", pc="PC-1"):
    return CloudEvent(event_id="{A-B-C}", ts=ts, principal="u", action=action,
                      resource=resource, source="file", pc=pc)


def test_vector_is_exactly_34_dimensional():
    p = PrincipalProfile("u")
    v = build_vector(_event(datetime(2010, 1, 1, 9)), p)
    assert v.shape == (N_FEATURES,) == (34,)
    assert len(FEATURE_NAMES) == 34
    assert np.isfinite(v).all()


def test_vector_is_34_dimensional_for_every_real_event(corpus):
    store = BaselineStore()
    for event in stream_events(corpus["root"]):
        profile = store.get(event.principal)
        v = build_vector(event, profile)
        assert v.shape == (34,)
        assert np.isfinite(v).all(), f"non-finite feature for {event.action}"
        observe_event(event, profile)


def test_cyclic_hours_wrap():
    """23:00 and 01:00 must be close; 00:00 and 12:00 must be far."""
    i_sin = FEATURE_NAMES.index("hour_sin")
    i_cos = FEATURE_NAMES.index("hour_cos")

    def circle(hour):
        v = build_vector(_event(datetime(2010, 1, 4, hour)), PrincipalProfile("u"))
        return np.array([v[i_sin], v[i_cos]])

    assert np.linalg.norm(circle(23) - circle(1)) < np.linalg.norm(circle(0) - circle(12))
    # The pair lies on the unit circle.
    assert np.linalg.norm(circle(7)) == pytest.approx(1.0)


def test_day_of_week_is_also_cyclic():
    i_sin, i_cos = FEATURE_NAMES.index("dow_sin"), FEATURE_NAMES.index("dow_cos")

    def circle(day):
        v = build_vector(_event(datetime(2010, 1, 4) + timedelta(days=day)),
                         PrincipalProfile("u"))
        return np.array([v[i_sin], v[i_cos]])

    # Sunday (6) and Monday (0) are adjacent.
    assert np.linalg.norm(circle(6) - circle(7)) < np.linalg.norm(circle(0) - circle(3))


def test_first_time_flags_flip_after_the_first_observation():
    p = PrincipalProfile("u")
    ts = datetime(2010, 1, 1, 9)
    i_action = FEATURE_NAMES.index("first_time_action")
    i_resource = FEATURE_NAMES.index("first_time_resource")

    e = _event(ts)
    v1 = build_vector(e, p)
    assert v1[i_action] == 1.0 and v1[i_resource] == 1.0
    observe_event(e, p)
    v2 = build_vector(_event(ts + timedelta(minutes=1)), p)
    assert v2[i_action] == 0.0 and v2[i_resource] == 0.0


def test_action_frequency_is_per_principal():
    """The whole point of the feature: identical action, different history, different value."""
    heavy, light = PrincipalProfile("heavy"), PrincipalProfile("light")
    t0 = datetime(2010, 1, 1, 9)
    for i in range(20):
        observe_event(_event(t0 + timedelta(minutes=i)), heavy)
    for i in range(20):
        observe_event(_event(t0 + timedelta(minutes=i), action="execute-api:Invoke"), light)

    idx = FEATURE_NAMES.index("action_frequency")
    probe = _event(t0 + timedelta(hours=1))
    assert build_vector(probe, heavy)[idx] > 0.9
    assert build_vector(probe, light)[idx] == pytest.approx(0.0)


def test_sensitivity_ranks_egress_highest():
    idx = FEATURE_NAMES.index("resource_sensitivity")
    removable = CloudEvent("{i}", datetime(2010, 1, 1), "u", "s3:GetObject",
                           "arn:aws:s3:::sentinel-x/removable/PC-1", "device", egress=True)
    session = _event(datetime(2010, 1, 1), action="sts:AssumeRole",
                     resource="arn:aws:iam::0:role/x")
    p = PrincipalProfile("u")
    assert build_vector(removable, p)[idx] > build_vector(session, p)[idx]


def test_rate_features_respond_to_a_burst():
    p = PrincipalProfile("u")
    t0 = datetime(2010, 1, 1, 9)
    idx = FEATURE_NAMES.index("calls_1min")
    quiet = build_vector(_event(t0), p)
    for i in range(30):
        observe_event(_event(t0 + timedelta(seconds=i)), p)
    burst = build_vector(_event(t0 + timedelta(seconds=30)), p)
    assert burst[idx] > quiet[idx]
    assert burst[idx] == 30


def test_network_features_are_deterministic_and_move_with_the_host():
    assert host_identity("PC-0001") == host_identity("PC-0001")
    assert host_identity("PC-0001") != host_identity("PC-0002")
    _, lat1, lon1 = host_identity("PC-0001")
    _, lat2, lon2 = host_identity("PC-0002")
    assert geodesic_km((lat1, lon1), (lat1, lon1)) == pytest.approx(0.0)
    assert geodesic_km((lat1, lon1), (lat2, lon2)) > 0
    assert geodesic_km(None, (lat2, lon2)) == 0.0


def test_no_future_leakage_in_a_single_vector():
    """A vector must reflect history strictly before its own event."""
    p = PrincipalProfile("u")
    t0 = datetime(2010, 1, 1, 9)
    idx = FEATURE_NAMES.index("log_action_history")
    first = build_vector(_event(t0), p)
    assert first[idx] == pytest.approx(0.0)   # no history yet
    observe_event(_event(t0), p)
    second = build_vector(_event(t0 + timedelta(minutes=1)), p)
    # Exactly one prior event, minus a minute of EWMA decay -- so just under log(2).
    assert second[idx] == pytest.approx(math.log1p(1.0), rel=1e-4)
    assert second[idx] < math.log1p(1.0)


# --- standardisation -------------------------------------------------------


def test_standardiser_touches_only_continuous_features():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(500, N_FEATURES)) * 10 + 5
    std = Standardiser.fit(matrix)
    out = std.transform(matrix)
    for i, name in enumerate(FEATURE_NAMES):
        if name in CONTINUOUS_FEATURES:
            assert abs(out[:, i].mean()) < 1e-9
            assert out[:, i].std() == pytest.approx(1.0)
        else:
            np.testing.assert_allclose(out[:, i], matrix[:, i])


def test_standardiser_survives_a_constant_column():
    matrix = np.ones((10, N_FEATURES))
    out = Standardiser.fit(matrix).transform(matrix)
    assert np.isfinite(out).all()


def test_standardiser_round_trips():
    rng = np.random.default_rng(1)
    std = Standardiser.fit(rng.normal(size=(100, N_FEATURES)))
    restored = Standardiser.from_dict(std.to_dict())
    np.testing.assert_allclose(restored.mean, std.mean)
    np.testing.assert_allclose(restored.std, std.std)


def test_streaming_standardiser_matches_batch_fit():
    """Welford accumulator must agree with Standardiser.fit, which it replaces on the
    full corpus because the training window does not fit in memory."""
    from sentinel.features.builder import StandardiserAccumulator

    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(2000, N_FEATURES)) * 7 + 3
    batch = Standardiser.fit(matrix)
    acc = StandardiserAccumulator()
    for row in matrix:
        acc.update(row)
    online = acc.finalize()
    assert len(acc) == 2000
    np.testing.assert_allclose(online.mean, batch.mean, atol=1e-9)
    np.testing.assert_allclose(online.std, batch.std, atol=1e-9)


def test_streaming_standardiser_memory_is_independent_of_rows():
    from sentinel.features.builder import StandardiserAccumulator

    acc = StandardiserAccumulator()
    assert not hasattr(acc, "__dict__"), "accumulator must stay __slots__-only"
    rng = np.random.default_rng(3)
    for _ in range(5000):
        acc.update(rng.normal(size=N_FEATURES))
    # Three fixed-size arrays regardless of how many rows were seen.
    assert acc._mean.shape == acc._m2.shape == (N_FEATURES,)


def test_streaming_standardiser_rejects_empty_input():
    from sentinel.features.builder import StandardiserAccumulator

    with pytest.raises(ValueError, match="no rows"):
        StandardiserAccumulator().finalize()


def test_labels_are_keyed_on_source_and_id(corpus):
    """r4.2 readme erratum: ids are unique per file, not globally."""
    labels = load_labels(corpus["root"] / "answers")
    eid = next(iter(corpus["malicious_ids"]))
    assert ("file", eid) in labels
    assert ("http", eid) not in labels        # same id, other file: not malicious
    assert labels.event_ids == corpus["malicious_ids"]
