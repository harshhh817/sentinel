"""Module 2 tests: detectors, calibration, fusion, trust algorithm, bands, and the
three scripts end to end on the synthetic 34-dim dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sentinel.config import (  # noqa: E402
    BANDS,
    FUSION_ALPHA,
    IFOREST_MAX_SAMPLES,
    IFOREST_N_ESTIMATORS,
)
from sentinel.features.builder import FEATURE_NAMES  # noqa: E402
from sentinel.risk.autoencoder import AEConfig, AutoEncoder, train_autoencoder  # noqa: E402
from sentinel.risk.fusion import EmpiricalCDF, RiskEngine, fuse  # noqa: E402
from sentinel.risk.iforest import isolation_score, train_iforest  # noqa: E402
from sentinel.risk.metrics import at_threshold, best_f1_threshold  # noqa: E402
from sentinel.risk.trust import DENY_THRESHOLD, band_lookup, effective_risk, verdicts  # noqa: E402
from tests.fixtures import write_synthetic_splits  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    truth = write_synthetic_splits(out, n_train=4000, n_val=1500, n_test=3000,
                                   anomaly_rate=0.03, seed=3)
    return truth


@pytest.fixture(scope="module")
def trained(synth, tmp_path_factory):
    from train import main as train_main

    models = tmp_path_factory.mktemp("models")
    rc = train_main(["--data", str(synth["out_dir"]), "--models", str(models),
                     "--seeds", "0", "1", "--epochs", "15", "--device", "cpu"])
    assert rc == 0
    return models


# --- autoencoder -----------------------------------------------------------


def test_autoencoder_architecture_is_34_24_16_8_mirrored():
    ae = AutoEncoder()
    lin_enc = [m for m in ae.encoder if isinstance(m, torch.nn.Linear)]
    lin_dec = [m for m in ae.decoder if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in lin_enc] == [(34, 24), (24, 16), (16, 8)]
    assert [(m.in_features, m.out_features) for m in lin_dec] == [(8, 16), (16, 24), (24, 34)]
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in ae.encoder)
    drops = [m for m in ae.encoder if isinstance(m, torch.nn.Dropout)]
    assert drops and all(d.p == 0.2 for d in drops)


def test_autoencoder_forward_shape_and_error_nonnegative():
    ae = AutoEncoder().eval()
    x = np.random.default_rng(0).normal(size=(50, 34)).astype(np.float32)
    assert ae(torch.as_tensor(x)).shape == (50, 34)
    e = ae.reconstruction_error(x)
    assert e.shape == (50,) and (e >= 0).all()


def test_autoencoder_learns_the_benign_manifold():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(3000, 4))
    w = rng.normal(size=(4, 34))
    x = np.tanh(z @ w).astype(np.float32)
    ae, log = train_autoencoder(x[:2500], x[2500:], config=AEConfig(epochs=60), seed=0,
                                device="cpu")
    assert log.val_loss[-1] < log.val_loss[0] * 0.6, "validation loss did not fall"
    # The real test: off-manifold points reconstruct much worse than held-out benign ones.
    off = rng.normal(size=(200, 34)).astype(np.float32) * 2
    assert ae.reconstruction_error(off).mean() > 3 * ae.reconstruction_error(x[2500:]).mean()


def test_early_stopping_restores_best_epoch():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 34)).astype(np.float32)
    _, log = train_autoencoder(x[:500], x[500:], config=AEConfig(epochs=200, patience=3),
                               seed=0, device="cpu")
    assert log.epochs_run < 200
    assert log.best_val_loss == min(log.val_loss)


# --- isolation forest ------------------------------------------------------


def test_iforest_hyperparameters_and_score_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 34)).astype(np.float32)
    f = train_iforest(x, seed=0)
    assert f.n_estimators == IFOREST_N_ESTIMATORS == 200
    assert f.max_samples == IFOREST_MAX_SAMPLES == 256
    s = isolation_score(f, x)
    assert ((s > 0) & (s <= 1)).all()
    outliers = rng.normal(size=(100, 34)).astype(np.float32) * 6 + 8
    assert isolation_score(f, outliers).mean() > s.mean()


# --- calibration and fusion ------------------------------------------------


def test_empirical_cdf_is_uniform_on_calibration_data_and_monotone():
    v = np.random.default_rng(0).exponential(size=5000)
    F = EmpiricalCDF(v)
    q = F(v)
    assert q.min() > 0 and q.max() == 1.0
    assert abs(q.mean() - 0.5) < 0.02
    grid = np.linspace(v.min(), v.max(), 100)
    assert (np.diff(F(grid)) >= 0).all()
    assert F(-1.0) == 0.0 and F(1e9) == 1.0


def test_fusion_weights():
    fe, fs = np.array([0.2, 0.9]), np.array([0.8, 0.1])
    np.testing.assert_allclose(fuse(fe, fs, 1.0), fe)
    np.testing.assert_allclose(fuse(fe, fs, 0.0), fs)
    np.testing.assert_allclose(fuse(fe, fs, FUSION_ALPHA), 0.6 * fe + 0.4 * fs)
    assert FUSION_ALPHA == 0.6


# --- trust algorithm (eq. 4) -----------------------------------------------


def test_R_is_zero_when_r_is_zero_for_any_s_and_c():
    for s in (0.0, 0.5, 1.0):
        for c in (0.0, 0.5, 1.0):
            assert effective_risk(0.0, s, c) == 0.0


def test_R_monotone_non_decreasing_in_r_and_s():
    r = np.linspace(0, 1, 201)
    for s in (0.0, 0.3, 1.0):
        assert (np.diff(effective_risk(r, s, 0.2)) >= -1e-12).all()
    s = np.linspace(0, 1, 201)
    for rv in (0.1, 0.5, 0.9):
        assert (np.diff(effective_risk(rv, s, 0.2)) >= -1e-12).all()


def test_credit_never_raises_risk_and_sensitivity_never_relaxes_it():
    assert effective_risk(0.5, 0.5, 1.0) < effective_risk(0.5, 0.5, 0.0)
    assert effective_risk(0.5, 1.0, 0.0) > effective_risk(0.5, 0.0, 0.0)
    assert effective_risk(1.0, 0.0, 0.0) == 1.0


def test_literal_form_reproduces_the_paper_typesetting_and_its_flaw():
    # As typeset, a perfectly normal request under full credit scores beta*c = 0.25.
    assert effective_risk(0.0, 0.5, 1.0, literal=True) == pytest.approx(0.25)
    assert effective_risk(0.0, 0.5, 1.0) == 0.0


def test_R_values_against_hand_computation():
    # R = [1 - (1-r)^(1+1.5 s)] (1 - 0.25 c)
    assert effective_risk(0.5, 1.0, 0.0) == pytest.approx(1 - 0.5 ** 2.5)
    assert effective_risk(0.5, 0.0, 1.0) == pytest.approx(0.5 * 0.75)


# --- bands (Table III) -----------------------------------------------------


@pytest.mark.parametrize("R,verdict", [
    (0.0, "ALLOW"), (0.3999, "ALLOW"), (0.40, "ALLOW_OBSERVE"), (0.6499, "ALLOW_OBSERVE"),
    (0.65, "STEPUP"), (0.8499, "STEPUP"), (0.85, "DENY"), (1.0, "DENY"),
])
def test_band_boundaries(R, verdict):
    assert band_lookup(R).verdict == verdict


def test_band_parameters_match_table_iii():
    assert (band_lookup(0.1).ttl_minutes, band_lookup(0.1).scope) == (60, "as_requested")
    assert (band_lookup(0.5).ttl_minutes, band_lookup(0.5).scope) == (30, "as_requested_verbose")
    assert (band_lookup(0.7).ttl_minutes, band_lookup(0.7).scope) == (15, "read_only")
    assert band_lookup(0.9).ttl_minutes is None
    assert DENY_THRESHOLD == 0.85 == BANDS[-1].lower


def test_vectorised_verdicts_agree_with_scalar_lookup():
    R = np.array([0.0, 0.4, 0.65, 0.85, 0.99, 0.3999])
    assert list(verdicts(R)) == [band_lookup(v).verdict for v in R]


# --- metrics ---------------------------------------------------------------


def test_metrics_at_threshold_and_tuned_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.95, 0.8])
    m = at_threshold(y, s, 0.85)
    assert (m.precision, m.recall) == (0.5, 0.5) and m.fpr == 0.25
    t = best_f1_threshold(y, s)
    assert at_threshold(y, s, t).f1 >= m.f1


# --- end to end ------------------------------------------------------------


def test_train_saves_a_loadable_engine_with_identical_scores(trained, synth):
    from sentinel.risk.data import load_split

    eng = RiskEngine.load(trained, 0)
    assert eng.feature_names == FEATURE_NAMES and eng.alpha == FUSION_ALPHA
    test = load_split(synth["out_dir"] / "test.parquet")
    r1 = eng.score(test.x).r
    r2 = RiskEngine.load(trained, 0).score(test.x).r
    np.testing.assert_allclose(r1, r2)
    assert ((r1 >= 0) & (r1 <= 1)).all()
    # Anomalies must score higher on average than benign rows.
    assert r1[test.y == 1].mean() > r1[test.y == 0].mean() + 0.2


def test_evaluate_writes_table_v_and_figures(trained, synth, tmp_path):
    from evaluate import evaluate

    out = tmp_path / "out"
    table = evaluate(synth["out_dir"], trained, out, seeds=[0, 1], device="cpu")
    assert set(table) == {"logistic_regression", "random_forest", "isolation_forest",
                          "autoencoder", "hybrid"}
    assert (out / "table_v.csv").exists() and (out / "fig3.png").exists()
    assert (out / "fig4.png").exists() and (out / "table_v_per_seed.json").exists()
    assert table["hybrid"]["auc"] > 0.9
    rows = (out / "table_v.csv").read_text().splitlines()
    assert len(rows) == 6 and rows[0].startswith("model,precision,recall,f1")


def test_ablation_writes_eight_rows(trained, synth, tmp_path):
    from ablation import ablation

    out = tmp_path / "abl"
    table = ablation(synth["out_dir"], trained, out, seeds=[0], epochs=5, subsample=4000,
                     val_subsample=1500, device="cpu")
    assert len(table) == 8
    rows = (out / "table_vi.csv").read_text().splitlines()
    assert len(rows) == 9
    assert "baseline" in rows[1]
    # alpha = 1 and alpha = 0 must be re-fusions of the same engine, not new models.
    assert table["no_iforest"]["auc"] > 0.5 and table["no_ae"]["auc"] > 0.5


def test_cert_replay_grants_no_control_credit():
    """CERT has no MFA/managed-device evidence, so c must be 0 (else R caps at 0.75)."""
    from sentinel.risk.data import control_credit_from_features

    x = np.random.default_rng(0).normal(size=(10, 34)).astype(np.float32)
    assert (control_credit_from_features(x) == 0).all()
    assert effective_risk(1.0, 0.0, 0.0) == 1.0          # deny is reachable


# --- per-source calibration and per-source models ----------------------------


def test_event_type_derivation():
    from sentinel.risk.types import event_type, event_type_one

    a = np.array(["sts:AssumeRole", "sts:SessionEnd", "s3:GetObject", "s3:GetObject",
                  "s3:PutObject", "s3:PutObject", "execute-api:Invoke"])
    s = np.array(["logon", "logon", "device", "pdp", "file", "pdp", "http"])
    assert list(event_type(a, s)) == ["sts", "sts", "egress", "s3_read", "egress", "s3_write",
                                      "http"]
    assert event_type_one("s3:GetObject", "pdp", "arn:aws:s3:::b/removable/x") == "egress"
    assert event_type_one("s3:GetObject", "pdp", "arn:aws:s3:::b/x") == "s3_read"


def test_split_loader_derives_types(synth):
    from sentinel.risk.data import load_split

    val = load_split(synth["out_dir"] / "val.parquet", with_types=True)
    assert val.types is not None and len(val.types) == len(val)
    assert set(val.types) <= {"sts", "s3_read", "s3_write", "egress", "http"}
    b = val.benign
    assert len(b.types) == len(b) and (b.y == 0).all()


def test_per_type_calibration_is_uniform_within_each_type(trained, synth):
    """The point of the change: benign r ~ U(0,1) inside every event type."""
    from sentinel.risk.data import load_split

    eng = RiskEngine.load(trained, 0)
    val = load_split(synth["out_dir"] / "val.parquet", with_types=True).benign
    e, s = eng.raw_scores(eng.standardise(val.x))
    eng.calibrate_per_type(e, s, val.types, min_rows=50)
    assert eng.calibration == "per_source" and set(eng.cdf_e_by_type) == set(np.unique(val.types))
    r = eng.score(val.x, types=val.types).r
    for typ in np.unique(val.types):
        m = val.types == typ
        assert abs(r[m].mean() - 0.5) < 0.08, f"{typ}: benign r mean {r[m].mean():.3f}"
    # Without types the global CDFs are used and results differ.
    assert not np.allclose(r, eng.score(val.x).r)


def test_per_source_calibration_survives_save_load(trained, synth, tmp_path):
    from sentinel.risk.data import load_split

    eng = RiskEngine.load(trained, 0)
    val = load_split(synth["out_dir"] / "val.parquet", with_types=True).benign
    e, s = eng.raw_scores(eng.standardise(val.x))
    eng.calibrate_per_type(e, s, val.types, min_rows=50)
    eng.save(tmp_path, 0)
    back = RiskEngine.load(tmp_path, 0)
    assert back.calibration == "per_source" and set(back.cdf_e_by_type) == set(eng.cdf_e_by_type)
    np.testing.assert_allclose(back.score(val.x, types=val.types).r,
                               eng.score(val.x, types=val.types).r)


def test_train_recalibrate_and_per_source_models(synth, tmp_path):
    from train import main as train_main

    from sentinel.risk.fusion import PerSourceEngine, load_engine

    m1 = tmp_path / "m1"
    assert train_main(["--data", str(synth["out_dir"]), "--models", str(m1), "--seeds", "0",
                       "--epochs", "5", "--device", "cpu"]) == 0
    assert load_engine(m1, 0).calibration == "global"
    assert train_main(["--data", str(synth["out_dir"]), "--models", str(m1), "--seeds", "0",
                       "--recalibrate", "--calibration", "per_source", "--device", "cpu"]) == 0
    assert load_engine(m1, 0).calibration == "per_source"

    m2 = tmp_path / "m2"
    assert train_main(["--data", str(synth["out_dir"]), "--models", str(m2), "--seeds", "0",
                       "--epochs", "5", "--per-source-models", "--device", "cpu"]) == 0
    eng = load_engine(m2, 0)
    assert isinstance(eng, PerSourceEngine) and eng.calibration == "per_source_models"
    from sentinel.risk.data import load_split

    test = load_split(synth["out_dir"] / "test.parquet", with_types=True)
    r = eng.score(test.x, types=test.types).r
    assert r.shape == (len(test),) and ((r >= 0) & (r <= 1)).all()
    assert r[test.y == 1].mean() > r[test.y == 0].mean()


def test_evaluate_and_compare_across_variants(trained, synth, tmp_path):
    from compare_variants import main as compare_main
    from evaluate import evaluate
    from train import main as train_main

    res = tmp_path / "results"
    evaluate(synth["out_dir"], trained, res / "global", seeds=[0], device="cpu",
             make_figures=False)
    m = tmp_path / "pscal"
    import shutil

    shutil.copytree(trained, m)
    assert train_main(["--data", str(synth["out_dir"]), "--models", str(m), "--seeds", "0",
                       "--recalibrate", "--calibration", "per_source", "--device", "cpu"]) == 0
    evaluate(synth["out_dir"], m, res / "per_source_calibration", seeds=[0], device="cpu",
             make_figures=False)
    assert compare_main(["--results", str(res)]) == 0
    rows = (res / "table_v_variants.csv").read_text().splitlines()
    assert rows[0].startswith("model,global:auc") and "per_source_calibration:auc" in rows[0]
    assert len(rows) == 6


# --- user-day aggregation ------------------------------------------------------


def test_userday_aggregation_counts_means_maxes_and_labels(synth, tmp_path):
    import pyarrow.parquet as pq

    from sentinel.risk.userday import N_USERDAY, USERDAY_FEATURES, aggregate, load, save

    path = synth["out_dir"] / "test.parquet"
    ud = aggregate(path)
    assert ud.x.shape == (len(ud), N_USERDAY) and len(USERDAY_FEATURES) == N_USERDAY
    assert np.isfinite(ud.x).all()
    # Reconcile against a pandas groupby on the same file.
    t = pq.read_table(path, columns=["principal", "ts", "label", "calls_60min"]).to_pandas()
    t["day"] = t["ts"].dt.strftime("%Y-%m-%d")
    g = t.groupby(["principal", "day"])
    ref_n = g.size()
    ref_max = g["calls_60min"].max()
    ref_lab = g["label"].max()
    k2i = ud.key_to_index()
    assert len(k2i) == len(ref_n)
    i_max = USERDAY_FEATURES.index("max_calls_60min")
    i_mean = USERDAY_FEATURES.index("mean_calls_60min")
    for (p, d), n in ref_n.items():
        i = k2i[(p, d)]
        assert ud.n_events[i] == n
        assert ud.x[i, i_max] == pytest.approx(ref_max[(p, d)], rel=1e-5)
        assert ud.x[i, i_mean] == pytest.approx(g["calls_60min"].mean()[(p, d)], rel=1e-4)
        assert ud.y[i] == ref_lab[(p, d)]
    assert ud.y.sum() > 0                                   # some malicious user-days
    save(ud, tmp_path / "ud.parquet")
    back = load(tmp_path / "ud.parquet")
    np.testing.assert_allclose(back.x, ud.x)
    assert list(back.principal) == list(ud.principal) and (back.y == ud.y).all()
