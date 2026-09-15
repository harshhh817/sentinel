#!/usr/bin/env python
"""Fig. 5: added authorisation latency by stage at a steady request rate -> --out/fig5.csv.

Drives the PDP in-process through an ASGI transport (no network, so the numbers are
the PDP's own cost), at ``--rps`` requests per second for ``--n`` requests, and
reports p50/p95 per stage from the per-request timings the endpoint returns.

    python scripts/latency_bench.py --models models --n 50000 --rps 200 --out results
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.config import MODELS, RESULTS  # noqa: E402
from sentinel.pdp.app import create_app  # noqa: E402
from sentinel.pdp.settings import Settings  # noqa: E402

STAGES = ("static_policy", "feature_assembly", "inference", "trust", "credential",
          "record_sign_enqueue", "baseline_update", "total")
ACTIONS = ["s3:GetObject", "s3:PutObject", "execute-api:Invoke", "sts:AssumeRole"]


def synthetic_request(rng: random.Random, i: int) -> dict:
    principal = f"U{rng.randrange(200):04d}"
    unit = "sales"
    action = rng.choice(ACTIONS)
    resource = {
        "s3:GetObject": f"arn:aws:s3:::sentinel-{unit}/docs/{rng.randrange(500)}.pdf",
        "s3:PutObject": f"arn:aws:s3:::sentinel-{unit}/docs/{rng.randrange(500)}.pdf",
        "execute-api:Invoke": f"arn:aws:execute-api:::site-{rng.randrange(50)}",
        "sts:AssumeRole": "arn:aws:iam::000000000000:role/sentinel-sales-session",
    }[action]
    return {"principal": principal, "action": action, "resource": resource,
            "context": {"device_id": f"PC-{rng.randrange(300):04d}", "device_managed": True,
                        "mfa_age_seconds": rng.uniform(0, 4 * 3600),
                        "mfa_hardware_backed": rng.random() < 0.5}}


async def run(models: Path, n: int, rps: float, seed: int, policy: Path | None,
              state_dir: Path, ledger: str = "jsonl") -> tuple[list[dict], dict]:
    import httpx

    state_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(models_dir=models, state_dir=state_dir, ledger=ledger)
    if policy:
        settings.policy_path = policy
    # Every synthetic principal belongs to "sales" so the own-bucket condition passes.
    doc = json.loads(settings.policy_path.read_text())
    doc["principals"].update({f"U{i:04d}": {"role": "user", "unit": "sales"} for i in range(200)})
    pol = state_dir / "bench_policy.json"
    pol.write_text(json.dumps(doc))
    settings.policy_path = pol

    app = create_app(settings)
    rng = random.Random(seed)
    timings: list[dict] = []
    verdicts: dict[str, int] = {}
    interval = 1.0 / rps
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://pdp") as client:
            t_start = time.perf_counter()
            for i in range(n):
                target = t_start + i * interval
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                resp = await client.post("/authorize", json=synthetic_request(rng, i))
                resp.raise_for_status()
                body = resp.json()
                timings.append(body["timings_ms"])
                verdicts[body["verdict"]] = verdicts.get(body["verdict"], 0) + 1
            elapsed = time.perf_counter() - t_start
        await app.state.pdp.queue.flush()
        committed = app.state.pdp.queue.committed
        rejected = app.state.pdp.queue.rejected
    return timings, {"requests": n, "target_rps": rps, "achieved_rps": round(n / elapsed, 1),
                     "elapsed_s": round(elapsed, 1), "verdicts": verdicts, "ledger": ledger,
                     "ledger_committed": committed, "ledger_rejected": rejected}


def summarise(timings: list[dict]) -> list[dict]:
    rows = []
    for stage in STAGES:
        v = np.array([t.get(stage, 0.0) for t in timings], dtype=np.float64)
        rows.append({"stage": stage, "p50_ms": round(float(np.percentile(v, 50)), 3),
                     "p95_ms": round(float(np.percentile(v, 95)), 3),
                     "mean_ms": round(float(v.mean()), 3)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--rps", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--state-dir", type=Path, default=None,
                    help="scratch dir for keys/ledger/evidence (default: a temp dir)")
    ap.add_argument("--ledger", choices=["jsonl", "sim", "fabric"], default="jsonl")
    a = ap.parse_args(argv)

    state = a.state_dir or Path(tempfile.mkdtemp(prefix="sentinel-bench-"))
    timings, meta = asyncio.run(run(a.models, a.n, a.rps, a.seed, a.policy, state, a.ledger))
    rows = summarise(timings)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "fig5.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "fig5_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    for r in rows:
        print(f"  {r['stage']:22s} p50 {r['p50_ms']:7.3f} ms   p95 {r['p95_ms']:7.3f} ms")
    print(f"wrote {a.out / 'fig5.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
