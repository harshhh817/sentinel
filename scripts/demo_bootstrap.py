#!/usr/bin/env python
"""Make `make demo` work from a fresh clone: synthesise what the real pipeline would produce.

If models/ and demo/ already hold the real artefacts (trained on CERT), this is a no-op.
Otherwise it generates a synthetic 34-dim dataset (tests/fixtures.py), trains one seed of
the request-level engine and the user-day models on it, and extracts a synthetic "insider"
scenario -- so the dashboard runs end to end with no data or disk. It says so in the sidebar.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.config import MODELS, ROOT  # noqa: E402

DEMO = ROOT / "demo"


def have_real_artefacts(models: Path = MODELS, demo: Path = DEMO) -> bool:
    return ((models / "engine_seed0.json").exists()
            and (models / "userday" / "random_forest_seed0.joblib").exists()
            and (demo / "scenario_meta.json").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="regenerate even if artefacts exist")
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--demo-dir", type=Path, default=DEMO)
    a = ap.parse_args(argv)
    models, demo = a.models, a.demo_dir
    if have_real_artefacts(models, demo) and not a.force:
        print("demo artefacts present; nothing to do")
        return 0
    sys.path.insert(0, str(ROOT))
    from demo_scenario import main as scenario_main
    from train import main as train_main
    from userday import main as userday_main

    from tests.fixtures import write_synthetic_splits

    work = Path(tempfile.mkdtemp(prefix="sentinel-demo-"))
    data = work / "data"
    print("synthesising a dataset ...", flush=True)
    write_synthetic_splits(data, n_train=20_000, n_val=6_000, n_test=12_000, anomaly_rate=0.03,
                           seed=7)
    print("training the request-level engine (1 seed) ...", flush=True)
    train_main(["--data", str(data), "--models", str(models), "--seeds", "0", "--epochs", "30",
                "--device", "cpu"])
    print("training the user-day models ...", flush=True)
    userday_main(["--data", str(data), "--models", str(models), "--out", str(work / "ud"),
                  "--seeds", "0", "--epochs", "30", "--device", "cpu", "--eval-subsample", "12000",
                  "--save-models", str(models / "userday")])
    print("extracting a synthetic insider scenario ...", flush=True)
    scenario_main(["--data", str(data), "--split", "test", "--out", str(demo)])
    meta = json.loads((demo / "scenario_meta.json").read_text())
    meta["synthetic"] = True
    (demo / "scenario_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"demo bootstrapped with SYNTHETIC data: scenario {meta['principal']}, "
          f"models in {models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
