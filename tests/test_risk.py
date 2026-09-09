"""Module 2 tests: detectors, calibration, fusion, trust algorithm, bands, and the
three scripts end to end on the synthetic 34-dim dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tests.fixtures import write_synthetic_splits  # noqa: E402
from ztb.config import BANDS, FUSION_ALPHA, IFOREST_MAX_SAMPLES, IFOREST_N_ESTIMATORS  # noqa: E402
from ztb.features.builder import FEATURE_NAMES  # noqa: E402
from ztb.risk.autoencoder import AEConfig, AutoEncoder, train_autoencoder  # noqa: E402
from ztb.risk.fusion import EmpiricalCDF, RiskEngine, fuse  # noqa: E402
from ztb.risk.iforest import isolation_score, train_iforest  # noqa: E402
from ztb.risk.metrics import at_threshold, best_f1_threshold  # noqa: E402
from ztb.risk.trust import DENY_THRESHOLD, band_lookup, effective_risk, verdicts  # noqa: E402


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
    from ztb.risk.data import load_split

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
    from ztb.risk.data import control_credit_from_features

    x = np.random.default_rng(0).normal(size=(10, 34)).astype(np.float32)
    assert (control_credit_from_features(x) == 0).all()
    assert effective_risk(1.0, 0.0, 0.0) == 1.0          # deny is reachable
