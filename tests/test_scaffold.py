"""Module 0 smoke tests: the scaffold imports and the paper's constants are what the paper says."""

import ztb
from ztb import config


def test_package_imports():
    assert ztb.__version__


def test_subpackages_import():
    import ztb.features  # noqa: F401
    import ztb.ledger  # noqa: F401
    import ztb.pdp  # noqa: F401
    import ztb.risk  # noqa: F401


def test_feature_vector_is_34_dimensional():
    """Table II. This invariant is re-tested against the real builder in Module 1."""
    assert config.N_FEATURES == 34
    assert sum(config.FEATURE_GROUPS.values()) == 34


def test_risk_engine_constants_match_paper():
    assert config.AE_ENCODER_WIDTHS == (34, 24, 16, 8)
    assert config.IFOREST_N_ESTIMATORS == 200
    assert config.IFOREST_MAX_SAMPLES == 256
    assert config.FUSION_ALPHA == 0.6
    assert config.TRUST_LAMBDA == 1.5
    assert config.TRUST_BETA == 0.25


def test_bands_cover_unit_interval_in_order():
    """Table III: four bands, ascending, starting at 0 and with the paper's boundaries."""
    lowers = [b.lower for b in config.BANDS]
    assert lowers == sorted(lowers)
    assert lowers == [0.00, 0.40, 0.65, 0.85]
    assert [b.verdict for b in config.BANDS] == ["ALLOW", "ALLOW_OBSERVE", "STEPUP", "DENY"]
    assert config.BANDS[-1].ttl_minutes is None  # deny issues no credential


def test_time_split_is_seventeen_months():
    """CERT r4.2 is 17 months of activity; the split must consume all of it."""
    assert sum(config.SPLIT_MONTHS.values()) == 17
