#!/usr/bin/env python
"""Fig. 6: committed throughput and commit latency against offered load -> --out/fig6.csv.

Submits signed records to the ledger at each offered rate for a fixed duration, from
``--workers`` concurrent submitters, and measures committed tx/s and per-transaction
commit latency (p50/p95). Meaningful only against a live Fabric network
(``--ledger fabric`` through the shim); ``--ledger sim`` exercises the harness
in-process and must not be reported as ledger throughput.

    python scripts/ledger_bench.py --ledger fabric --rates 50 100 200 300 400 500 --out results
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tamper_test import make_record  # noqa: E402

from sentinel.config import RESULTS  # noqa: E402
from sentinel.ledger.sim import LedgerRejected, SimLedger  # noqa: E402
from sentinel.pdp.chain import GENESIS, record_hash  # noqa: E402
from sentinel.pdp.signer import Signer, Verifier  # noqa: E402


class Principals:
    """Per-principal chain heads shared by the submitters (what the PDP tracks)."""

    TAG = datetime.now().strftime("%H%M%S")      # unique principals per run on a reused channel

    def __init__(self, signer: Signer, n: int, offset: int = 0):
        self.signer = signer
        self.heads = {f"B{self.TAG}-{offset + i:05d}": (0, GENESIS) for i in range(n)}
        self.lock = threading.Lock()

    def next_record(self, rng: random.Random, owned: list[str] | None = None) -> dict:
        """Next record for one of ``owned`` principals (a worker's disjoint share), so no
        two in-flight submissions carry consecutive seqs of the same chain."""
        with self.lock:
            p = rng.choice(owned or list(self.heads))
            seq, prev = self.heads[p]
            rec = make_record(self.signer, p, seq + 1, prev, datetime.now(UTC))
            self.heads[p] = (seq + 1, record_hash(rec))
            return rec


def run_rate(ledger, principals: Principals, rate: float, seconds: float, workers: int,
             seed: int) -> dict:
    interval = workers / rate
    latencies: list[float] = []
    committed = rejected = failed = 0
    lock = threading.Lock()
    stop = time.perf_counter() + seconds

    names = list(principals.heads)

    def worker(k: int):
        nonlocal committed, rejected, failed
        rng = random.Random(seed * 1000 + k)
        owned = names[k::workers]                 # disjoint principals per worker
        nxt = time.perf_counter() + k * interval / workers
        while True:
            now = time.perf_counter()
            if now >= stop:
                return
            if now < nxt:
                time.sleep(min(nxt - now, 0.005))
                continue
            nxt += interval
            rec = principals.next_record(rng, owned)
            t0 = time.perf_counter()
            try:
                ledger.commit(rec)
                ok = "c"
            except LedgerRejected:
                ok = "r"
            except Exception:  # noqa: BLE001
                ok = "f"
            dt = time.perf_counter() - t0
            with lock:
                if ok == "c":
                    committed += 1
                    latencies.append(dt)
                elif ok == "r":
                    rejected += 1
                else:
                    failed += 1

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(workers)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start
    lat = np.array(latencies) * 1000 if latencies else np.array([np.nan])
    return {"offered_tps": rate, "committed_tps": round(committed / elapsed, 1),
            "committed": committed, "rejected": rejected, "failed": failed,
            "commit_p50_ms": round(float(np.nanpercentile(lat, 50)), 1),
            "commit_p95_ms": round(float(np.nanpercentile(lat, 95)), 1),
            "seconds": round(elapsed, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", choices=["sim", "fabric"], default="fabric")
    ap.add_argument("--shim", default=None)
    ap.add_argument("--rates", type=float, nargs="+", default=[50, 100, 200, 300, 400, 500, 600])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--principals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)

    pdp = Signer.load_or_create(a.keys_dir) if a.keys_dir else Signer.generate()
    if a.ledger == "sim":
        ledger = SimLedger(Verifier.from_signer(pdp))
    else:
        from sentinel.ledger.client import FabricLedger

        ledger = FabricLedger(a.shim) if a.shim else FabricLedger()
        try:
            ledger.set_pdp_public_key(pdp.public_pem().decode())
        except Exception as e:  # noqa: BLE001
            if "already set" not in str(e) or not a.keys_dir:
                raise SystemExit(f"cannot install the PDP key: {e}") from e
    rows = []
    for i, rate in enumerate(a.rates):
        principals = Principals(pdp, a.principals, offset=i * a.principals)
        row = run_rate(ledger, principals, rate, a.seconds, a.workers, a.seed)
        row["ledger"] = a.ledger
        rows.append(row)
        print(f"  offered {rate:6.0f} tx/s -> committed {row['committed_tps']:6.1f} tx/s  "
              f"p50 {row['commit_p50_ms']:7.1f} ms  p95 {row['commit_p95_ms']:7.1f} ms  "
              f"rejected {row['rejected']} failed {row['failed']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "fig6.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "fig6_meta.json").write_text(json.dumps({"ledger": a.ledger, "seconds": a.seconds,
                                                     "workers": a.workers}, indent=2))
    print(f"wrote {a.out / 'fig6.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
