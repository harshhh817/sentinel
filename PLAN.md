# Sentinel — Build Plan

(Sentinel implements the ZTBAudit architecture from the paper.)

Four modules, in order. Each ends with passing tests and a short report. Do not start the next module until told to.

---

## Module 0 — Scaffold (½ day)
- [ ] Create repo layout from CLAUDE.md, `requirements.txt`, `.gitignore` (data/, *.pt, .env), `README.md` stub.
- [ ] `make setup`, `make test`, `make train`, `make eval` targets.
- [ ] Git init, set author, first commit.

## Module 1 — Data + Features (Week 1)
**Input:** CERT r4.2 CSVs (logon.csv, file.csv, device.csv, http.csv, LDAP/, answers/).
**Deliverables:**
- [ ] `sentinel/features/cert_mapper.py` — maps CERT rows to CloudTrail-style JSON events per the mapping in CLAUDE.md. Preserves timestamps, user IDs, OU → bucket assignment.
- [ ] `sentinel/features/labels.py` — attaches malicious labels from `answers/` (scenario files).
- [ ] `sentinel/features/baseline.py` — per-principal profile store (dict/SQLite locally, DynamoDB in cloud mode), EWMA update with 30-day half-life.
- [ ] `sentinel/features/builder.py` — builds the 34-dim vector for one event given the principal's baseline and the sliding-window state. Standardisation stats fitted on train window only.
- [ ] `scripts/build_dataset.py` — streams events in time order, produces `data/processed/{train,val,test}.parquet` with time-based split (10/2/5 months).
- [ ] Tests: vector length == 34, sin/cos encoding, window anchoring, no future leakage in split.
**Report:** event counts per split, positive rate on test (~0.18 %), feature summary stats.

## Module 2 — Risk Engine + Trust Algorithm (Week 2)
- [ ] `sentinel/risk/autoencoder.py` — PyTorch model 34-24-16-8 mirrored; train benign only; save `models/ae.pt`.
- [ ] `sentinel/risk/iforest.py` — sklearn IsolationForest(200, 256); save pickle.
- [ ] `sentinel/risk/fusion.py` — empirical CDF calibration on held-out benign set; `score(x) -> r`.
- [ ] `sentinel/risk/trust.py` — eq. (4) + Table III `band_lookup(R) -> (verdict, ttl, scope)`.
- [ ] `scripts/train.py` (5 seeds), `scripts/evaluate.py` → `results/table_v.csv`, ROC plot, PR bar chart.
- [ ] `scripts/ablation.py` → `results/table_vi.csv` (7 variants).
- [ ] Baselines: logistic regression, random forest (supervised), iForest-only, AE-only.
- [ ] Tests: R monotone in r and s; r == 0 ⇒ R == 0 for any s; band boundaries.
**Report:** Table V and VI reproduced; note where numbers differ from the paper and why.

## Module 3 — PEP / PDP Service (Week 3)
- [ ] `sentinel/pdp/app.py` — FastAPI. `POST /authorize` implements Algorithm 1 exactly (static check → features → score → trust → verdict → credential → record → async enqueue → baseline update).
- [ ] `sentinel/pdp/static_policy.py` — simple RBAC/ABAC JSON policy evaluator (stand-in for IAM in local mode).
- [ ] `sentinel/pdp/credentials.py` — local mode: signed mock token with TTL/scope; cloud mode: `sts.assume_role` with session policy.
- [ ] `sentinel/pdp/signer.py` — ECDSA P-256 key gen, sign/verify over canonical JSON of the record tuple.
- [ ] `sentinel/pdp/chain.py` — per-principal seq + prevHash; `h_feat = sha256(x || salt)`; off-chain evidence store (encrypted file).
- [ ] `sentinel/pdp/queue.py` — in-memory/asyncio queue + committer worker (ledger client plugged in Module 4; until then, append-only JSONL).
- [ ] `scripts/latency_bench.py` — 50k requests at 200 rps, per-stage p50/p95 → `results/fig5.csv`.
- [ ] Tests: end-to-end allow/step-up/deny paths; signature tamper rejected; chain continuity.
**Report:** latency breakdown, sample audit records, demo curl commands.

## Module 4 — Ledger + Tamper Experiment (Week 4)
- [ ] Use `fabric-samples/test-network` (Docker). Document setup in `chaincode/README.md`.
- [ ] `chaincode/auditcontract/` (Go): `LogAccess` (verify sig, seq continuity, reject dup recId), `QueryByPrincipal`, `QueryByResource` (CouchDB rich queries), `VerifyChain`. No update/delete.
- [ ] `sentinel/ledger/client.py` — Fabric Gateway SDK (Python via gRPC or thin Go/Node shim); committer drains queue → `LogAccess`.
- [ ] `scripts/tamper_test.py` — 500 attempts (delete / modify verdict / backdate / fabricate), mirrored against a plain JSONL log vs the ledger; 10k clean control run → `results/table_vii.csv`.
- [ ] `scripts/ledger_bench.py` — offered load vs committed tx/s and commit latency → `results/fig6.csv`.
- [ ] Tests: VerifyChain returns first discontinuity index; fabricated record rejected.
**Report:** 500/500 detection, 0 false alarms, throughput curve.

## Module 5 (optional) — AWS Cloud Mode
- [ ] `infra/` SAM template: API Gateway HTTP API + Lambda authorizer (PEP) + Lambda PDP + SageMaker endpoint (or Lambda-hosted model to save cost) + DynamoDB baseline + STS role with session policies.
- [ ] Deploy to ap-south-1, run a 1-hour smoke test, tear down. Record cost.

## Demo / Viva checklist
- [ ] `make demo`: starts PDP + test-network, replays 200 events, shows allow/step-up/deny + ledger entries + a live tamper → VerifyChain catch.
- [ ] `results/` regenerated from scratch by `make eval`.
- [ ] README with architecture diagram, setup, and how each paper table maps to a script.
