# Sentinel — Project Spec

Sentinel implements the **ZTBAudit** architecture from the paper below; the project, package and all user-facing names are *Sentinel*, and "ZTBAudit" is used only when referring to the paper.

Implementation of the paper **"AI-Driven Zero-Trust Cloud Access Control with Blockchain-Anchored Audit Trails: An AWS-Based Architecture"** (Gupta, Shivam, Aditya, Naaz, Kumar — Sharda University). The PDF is in this folder; read it before any non-trivial change.

## Goal
Working, demo-able prototype of the ZTBAudit architecture (as *Sentinel*) that reproduces the paper's pipeline end to end on a laptop (local mode) with an optional AWS deployment (cloud mode). Final-year B.Tech project — correctness and a clean demo matter more than scale.

## Owner context
- Team: Harsh Gupta (lead), Shivam, Aditya. Guides: Sheenam Naaz, Kapil Kumar.
- Languages: Python 3.11 (ML, PDP/PEP), Go (Fabric chaincode). Java is NOT used here.
- Budget: minimise AWS spend. Everything must run locally first; AWS is a stretch goal.

## Architecture (from the paper)
Request → **PEP** (auth + device posture) → **PDP** (features → risk → trust algorithm → verdict) → **STS scoped credential** (cloud mode) or mock token (local mode) → **signed audit record** queued async → **Hyperledger Fabric** ledger.

### Feature vector — 34 dims (Table II)
| Group | Dims | Examples |
|---|---|---|
| Identity & entitlement | 5 | principal type, role tenure, policy breadth, cross-account depth |
| Temporal | 6 | hour sin/cos, day-of-week, deviation from modal hours, inter-request gap |
| Action | 7 | per-principal action frequency, read/write ratio, first-time flag, action-class entropy |
| Resource | 5 | sensitivity label, cross-account flag, resource novelty, ARN prefix depth |
| Volume & rate | 6 | calls in 1/15/60-min sliding windows, bytes read/60min, distinct resources |
| Network & device | 5 | ASN novelty, geodesic distance from prev request, device-fingerprint match, MFA age |

Rules: standardise continuous features; sin/cos for cyclic; per-principal empirical frequency for high-cardinality categoricals; sliding windows anchored on request timestamp (not wall-clock buckets); baseline updated with EWMA, 30-day effective half-life.

### Risk engine (Section IV-C)
- **Autoencoder**: encoder 34-24-16-8, mirrored decoder, ReLU, BatchNorm, dropout 0.2. Trained on benign only. Loss = mean ‖x − g(f(x))‖². Score e(x) = reconstruction L2 error.
- **Isolation forest**: 200 trees, subsample 256. Score s(x) per eq. (2).
- **Fusion** (eq. 3): r = α·F̂ₑ(e) + (1−α)·F̂ₛ(s), where F̂ are empirical CDFs fitted on a held-out benign calibration set. α = 0.6.
- Training: Adam lr 1e-3, batch 512, 120 epochs, cosine annealing, early stopping patience 10. Mean over 5 seeds.

### Trust algorithm (eq. 4)
R = 1 − (1 − r)^(1 + λs) · (1 − βc), with λ = 1.5, β = 0.25, s = resource sensitivity ∈ [0,1], c = compensating-control credit ∈ [0,1].
Static RBAC/ABAC entitlement is checked FIRST; if IAM forbids, DENY regardless of R.

### Verdict bands (Table III)
| R | Verdict | TTL | Scope |
|---|---|---|---|
| < 0.40 | ALLOW | 60 min | as requested |
| 0.40–0.65 | ALLOW (observe) | 30 min | as requested + verbose logging |
| 0.65–0.85 | STEP-UP | 15 min | read-only where possible |
| ≥ 0.85 | DENY | — | — |

### Audit record (eq. 5)
`⟨recId, prevHash, seq, ts, principal, action, resource, r, R, verdict, h_feat, sig⟩`
- seq: per-principal monotonic counter; prevHash: SHA-256 of previous record for that principal.
- h_feat = SHA-256(x ‖ salt); raw x + salt stored off-chain (local: encrypted file; cloud: S3 Object Lock).
- sig: ECDSA P-256 over the tuple by the PDP key.
- Chaincode (Go): `LogAccess`, `QueryByPrincipal`, `QueryByResource`, `VerifyChain`. NO update/delete transactions.

### Data
CERT r4.2 insider-threat corpus replayed as CloudTrail-style events. Mapping: logon/logoff → sts:AssumeRole / session end; file events → s3:GetObject / s3:PutObject on the user's OU bucket; removable-device → s3:GetObject with egress marker; web → external proxy. Keep original timestamps, identities, org structure.
Split BY TIME: first 10 months train (benign only), next 2 validation, last 5 test. Never random split.

### Targets to reproduce (Table V–VII)
Hybrid F1 ≈ 0.93, AUC ≈ 0.96, FPR ≈ 0.4 % at R ≥ 0.85; ablations (no iForest, no AE, global vs per-principal action freq, no rate features, α = 0.4/0.8); median added latency; 500/500 tamper attempts detected, 0 false alarms on 10k clean records.

## Repo layout
```
sentinel/
  CLAUDE.md  PLAN.md  README.md  paper.pdf
  data/            raw CERT (gitignored), processed parquet
  sentinel/
    features/      CERT→CloudTrail mapper, feature builder, baseline store
    risk/          autoencoder.py, iforest.py, fusion.py, trust.py
    pdp/           FastAPI app implementing Algorithm 1, signer, hash chain, queue
    ledger/        fabric client, committer, verify tooling
  chaincode/       Go: auditContract
  scripts/         train.py, evaluate.py, ablation.py, latency_bench.py, tamper_test.py
  infra/           (cloud mode) SAM/Terraform for API GW + Lambda + SageMaker + STS
  tests/
  results/         tables + figures generated by scripts
```

## Working rules
- Follow PLAN.md; complete one module, run its tests, then stop and report before moving on.
- Local mode is the default; never introduce an AWS dependency without a local fallback.
- Every number in `results/` must be produced by a script, not hand-typed.
- Write pytest tests for: feature dimensions (must be exactly 34), band lookup boundaries, monotonicity of R in r and s, hash-chain continuity, signature rejection.
- Use `uv` or `pip` with a pinned `requirements.txt`. Python 3.11, PyTorch 2.x, scikit-learn 1.4+.
- Keep explanations short; Harsh prefers crisp, plain-language summaries.
- Commit after each module with a descriptive message. Git author must be set to Harsh's real name/email (not "Your Name").
