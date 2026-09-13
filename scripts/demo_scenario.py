#!/usr/bin/env python
"""Extract one CERT insider's test-window events for the demo replay -> demo/.

Ranks insiders by malicious events in the test window, picks the top one (or
``--principal``), and writes that principal's requests in time order with the 34
features, plus the same principal's user-days, so the dashboard can replay them
day by day without touching the 7.7 M-row split.

    python scripts/demo_scenario.py --data <splits> --out demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ztb.config import DATA_PROCESSED, ROOT  # noqa: E402


def rank_insiders(path: Path, top: int = 8) -> list[dict]:
    t = pq.read_table(path, columns=["principal", "label", "source", "ts"],
                      filters=[("label", "==", 1)])
    if t.num_rows == 0:
        return []
    df = t.to_pandas()
    g = df.groupby("principal")
    rows = []
    for p, sub in g:
        rows.append({"principal": p, "malicious_events": int(len(sub)),
                     "days": int(sub["ts"].dt.normalize().nunique()),
                     "first": sub["ts"].min().strftime("%Y-%m-%d"),
                     "last": sub["ts"].max().strftime("%Y-%m-%d"),
                     "sources": dict(sub["source"].value_counts())})
    rows.sort(key=lambda r: -r["malicious_events"])
    return rows[:top]


def extract(path: Path, principal: str, out: Path, context_days: int = 3) -> dict:
    t = pq.read_table(path, filters=[("principal", "==", principal)])
    t = t.sort_by("ts")
    # keep the scenario window plus a few days of context either side
    lab = t.column("label").to_numpy()
    ts = t.column("ts").to_pandas()
    mal = ts[lab == 1]
    lo, hi = mal.min().normalize(), mal.max().normalize()
    import pandas as pd

    lo -= pd.Timedelta(days=context_days)
    hi += pd.Timedelta(days=context_days + 1)
    mask = pa.array((ts >= lo) & (ts < hi))
    t = t.filter(mask)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, out / f"scenario_{principal}.parquet")
    meta = {"principal": principal, "events": t.num_rows,
            "malicious": int(pc.sum(t.column("label")).as_py()),
            "window": [str(lo.date()), str((hi - pd.Timedelta(days=1)).date())],
            "days": int(t.column("ts").to_pandas().dt.normalize().nunique())}
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--split", default="test")
    ap.add_argument("--principal", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "demo")
    a = ap.parse_args(argv)
    path = a.data / f"{a.split}.parquet"
    ranked = rank_insiders(path)
    for r in ranked:
        print(f"  {r['principal']}  malicious {r['malicious_events']:>5}  days {r['days']:>3}  "
              f"{r['first']}..{r['last']}  {r['sources']}")
    principal = a.principal or ranked[0]["principal"]
    meta = extract(path, principal, a.out)
    meta["candidates"] = ranked
    (a.out / "scenario_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"scenario: {meta['principal']}  {meta['events']:,} events "
          f"({meta['malicious']} malicious) over {meta['days']} days {meta['window']} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
