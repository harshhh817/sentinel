"""Module 1 tests for the split: time-ordered, non-overlapping, no future leakage."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_dataset import boundaries_from, build, corpus_start  # noqa: E402

from tests.fixtures import write_corpus  # noqa: E402
from ztb.config import SPLIT_MONTHS  # noqa: E402
from ztb.features.builder import FEATURE_NAMES  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = write_corpus(tmp_path_factory.mktemp("cert") / "r4.2", days=400, events_per_day=6)
    out = tmp_path_factory.mktemp("processed")
    report = build(root["root"], out, progress_every=0)
    return {"corpus": root, "out": out, "report": report}


def _read(out: Path, split: str):
    return pq.read_table(out / f"{split}.parquet").to_pydict()


def test_all_three_splits_are_written(built):
    for split in ("train", "val", "test"):
        assert (built["out"] / f"{split}.parquet").exists()
        assert built["report"]["splits"][split]["rows"] > 0


def test_split_boundaries_follow_the_paper():
    b = boundaries_from(datetime(2010, 1, 1))
    train_days = (b.train_end - b.start).days
    val_days = (b.val_end - b.train_end).days
    assert round(train_days / 30.44) == SPLIT_MONTHS["train"] == 10
    assert round(val_days / 30.44) == SPLIT_MONTHS["val"] == 2


def test_corpus_start_is_the_earliest_event(built):
    assert corpus_start(built["corpus"]["root"]) == built["corpus"]["start"]


def test_splits_are_disjoint_in_time(built):
    """The core no-leakage property: every train row predates every val row, and so on."""
    train, val, test = (_read(built["out"], s) for s in ("train", "val", "test"))
    assert max(train["ts"]) < min(val["ts"])
    assert max(val["ts"]) < min(test["ts"])


def test_splits_are_disjoint_in_events(built):
    ids = [set(_read(built["out"], s)["event_id"]) for s in ("train", "val", "test")]
    assert ids[0] & ids[1] == set()
    assert ids[1] & ids[2] == set()
    assert ids[0] & ids[2] == set()


def test_training_window_is_benign_only(built):
    """Section V: both detectors are unsupervised and never see a scripted scenario."""
    train = _read(built["out"], "train")
    assert set(train["label"]) == {0}


def test_malicious_events_land_in_the_held_out_windows(built):
    report = built["report"]
    positives = sum(report["splits"][s]["positives"] for s in ("val", "test"))
    dropped = report["train_malicious_dropped"]
    assert positives + dropped == len(built["corpus"]["malicious_ids"])
    assert positives > 0, "fixture should place malicious events after the train window"


def test_every_row_carries_all_34_features(built):
    table = pq.read_table(built["out"] / "test.parquet")
    for name in FEATURE_NAMES:
        assert name in table.column_names
    assert len([c for c in table.column_names if c in set(FEATURE_NAMES)]) == 34


def test_rows_are_written_in_timestamp_order(built):
    for split in ("train", "val", "test"):
        stamps = _read(built["out"], split)["ts"]
        assert stamps == sorted(stamps)


def test_standardiser_is_fitted_on_train_only(built):
    assert built["report"]["standardiser_fitted_on"] == "train"


def test_report_counts_reconcile(built):
    report = built["report"]
    written = sum(report["splits"][s]["rows"] for s in ("train", "val", "test"))
    assert written + report["train_malicious_dropped"] == report["events_mapped"]
    assert report["events_mapped"] == built["corpus"]["total_events"]


def test_report_is_persisted(built):
    assert (built["out"] / "report.json").exists()


def test_sample_mode_covers_all_three_splits(tmp_path):
    """The 1-month validation run must exercise every branch, not just train."""
    root = write_corpus(tmp_path / "r4.2", days=40, events_per_day=8)
    out = tmp_path / "sample"
    report = build(root["root"], out, months=1, progress_every=0)
    assert report["months_sampled"] == 1
    assert all(report["splits"][s]["rows"] > 0 for s in ("train", "val", "test"))
    assert report["events_mapped"] < root["total_events"], "sample should not read the whole corpus"
