# ZTBAudit

Reference implementation of **"AI-Driven Zero-Trust Cloud Access Control with Blockchain-Anchored
Audit Trails: An AWS-Based Architecture"** — Harsh Gupta, Shivam, Aditya, Sheenam Naaz, Kapil Kumar
(Sharda University). The paper is `paper.pdf` in this repo. Final-year B.Tech project: every
component of the paper is built, every reported number is reproduced by a script in `results/`,
and where the paper's numbers do not reproduce the README says so and why.

```mermaid
flowchart LR
    subgraph Request path — synchronous
        S[Subject] --> PEP[PEP<br/>auth + device posture]
        PEP --> SP{Static RBAC/ABAC<br/>IAM stand-in}
        SP -- forbidden --> DENY[DENY]
        SP -- permitted --> F[34-dim feature vector<br/>per-principal baseline, EWMA 30 d]
        F --> RE[Risk engine<br/>autoencoder + isolation forest<br/>r = α F̂ₑ + (1−α) F̂ₛ]
        F --> UD[User-day RF<br/>operating configuration]
        RE --> T[Trust algorithm eq. 4<br/>R = f(r, sensitivity, MFA credit)]
        UD --> T
        T --> V{Table III band}
        V -- R<0.40 --> A[ALLOW 60 min]
        V -- 0.40–0.65 --> AO[ALLOW observe 30 min]
        V -- 0.65–0.85 --> SU[STEP-UP 15 min<br/>signed challenge]
        V -- ≥0.85 --> DENY
        A & AO & SU --> CRED[Scoped credential<br/>mock token / STS]
    end
    subgraph Evidence path — asynchronous
        T --> REC[Signed record eq. 5<br/>seq, prevHash, h_feat, ECDSA P-256]
        REC --> Q[Queue + committer]
        Q --> L[(Hyperledger Fabric<br/>auditcontract: LogAccess, QueryBy*, VerifyChain)]
        REC --> EV[(Encrypted off-chain<br/>evidence store x ‖ salt)]
    end
```

**Quick start:** `make setup && make test` — then `make demo` (see *How to run the demo* below).

## Results vs paper

Every number below is produced by a script; the paper's are from its Tables V–VII and Figs. 5–6.

| Quantity | Paper | This implementation | Why the gap |
|---|---:|---:|---|
| Test-window events / positives | 2,184,663 / 3,912 (0.179 %) | 7,675,354 / 1,754 (0.023 %) | We replay all four CERT sources at event level (94 % http); the paper's smaller window implies http was not replayed per event. |
| Hybrid AUC (paper's design) | 0.964 | 0.748 | 94 % http makes the benign calibration http-shaped; the 0.748 is mostly "is not http", a class-prior artefact. |
| Hybrid F1 / FPR at R ≥ 0.85 | 0.927 / 0.41 % | 0.001 / 72 % | A benign-quantile r puts ~35 % of benign traffic above R ≥ 0.85 by construction; the paper's operating point cannot hold for a quantile score. |
| Hybrid AUC, per-source calibration | — | 0.481 | Removing the artefact reveals chance-level within-type signal on the test window. |
| Hybrid AUC, per-source models | — | 0.525 | Same; the autoencoder stays below 0.5 (malicious rows reconstruct slightly *better*). |
| Isolation forest / autoencoder AUC | 0.897 / 0.941 | 0.774 / 0.680 | Same mechanism; the autoencoder is the weaker detector here, the reverse of the paper. |
| Supervised RF AUC (event level) | 0.913 | 0.997 | The scenarios leave real, discriminative structure in the 34 features; it is not off-manifold. |
| Unsupervised hybrid AUC, user-day | — | 0.800 (forest 0.791) | Aggregation is where the unsupervised signal lives; propagated to requests: 0.748 → 0.799. |
| **Operating configuration** | hybrid | **RF on user-day vectors, AUC 0.954 (LR) / 0.911 (RF)** | The only configurations above 0.85; supervised, labelled from the validation window's first half (training window rerun pending the disk). |
| Ablation: drop autoencoder / drop forest | −0.087 / −0.033 F1 | AUC 0.774 / 0.680 (vs 0.748) | Dropping the autoencoder *helps*; per-principal action frequency still beats global (0.748 vs 0.733). |
| Injected gross anomalies (sanity) | — | hybrid AUC 0.997 (3 AM egress, 50× volume) | The pipeline detects off-manifold behaviour; the scripted scenarios are not off-manifold at event level. |
| Added latency, median / p95 | 61.3 / 109.8 ms | 2.57 / 2.79 ms | In-process PDP with the model co-located; the paper's figure includes API Gateway → Lambda → SageMaker hops. |
| Ledger peak throughput | ≈ 450 tx/s | ≈ 50 tx/s | Nine t3.medium hosts and a three-node Raft orderer vs one laptop running two peers + one orderer under Docker Desktop, 2 s batch timeout. |
| Commit latency at 200 tx/s | 84 ms | 563 ms p50 | Same; block cadence dominates on the laptop network. |
| Tampering detected / false alarms | 500 / 500, 0 / 10⁴ | **500 / 500, 0 / 10,000** (live Fabric) | Reproduced; edits applied directly to the peer's CouchDB, fabrication rejected at endorsement. |

Three departures from the paper's text are deliberate and documented in the source: eq. (4) as
typeset makes the compensating-control credit *raise* risk (R = βc at r = 0), so the credit scales
risk down instead; c = 0 throughout the CERT replay (the corpus has no MFA/managed-device signal;
the PDP computes c from a real auth context); the supervised baselines train on the scenario rows
that fall inside the training window (`train_malicious.parquet`).

## How to run the demo (fresh clone)

```bash
git clone <repo> ztbaudit && cd ztbaudit
make setup                                   # Python 3.11 venv, pinned deps
make test                                    # 44 tests, no network needed
# models: either train (needs data/processed from `make dataset`) or unpack a models/ bundle
make train && make eval                      # ~2.5 h on a laptop; or copy models/ from a teammate
python scripts/userday.py --data data/processed --save-models models/userday   # user-day RF (SHAP)
make demo-scenario                           # one insider's test-window events -> demo/
make demo-check                              # Docker, network, shim, models, scenario
make demo                                    # Streamlit at http://localhost:8501
```

Live-ledger mode (optional, ~15 min the first time): install Docker Desktop, then follow
`chaincode/README.md` — `install-fabric.sh`, `network.sh up createChannel -ca -s couchdb`, build the
chaincode image, `deployCCAAS`, disable the peers' state cache, `npm install && node server.js` in
`ztb/ledger/shim`. `make demo-check` reports which of these is missing; without them the demo
falls back to the in-process reference ledger with the same behaviour. `docs/DEMO_SCRIPT.md` is
the five-minute runbook; `docs/VIVA.md` the examiner Q&A.

## Status

| Module | Scope | State |
|---|---|---|
| 0 | Scaffold, dependencies, Makefile | **done** |
| 1 | CERT → CloudTrail mapping, 34-dim feature builder, time split | **done** |
| 2 | Risk engine, trust algorithm, Tables V and VI | **done — see Results** |
| 3 | PEP/PDP FastAPI service, signing, hash chain, Fig. 5 | **done (local mode)** |
| 4 | Fabric chaincode, committer, tamper experiment, Table VII | **done (chaincode tested; live network needs Docker)** |
| 3b | Demo dashboard: Plain IAM vs ZTBAudit, one insider replayed day by day | **done** |
| 5 | AWS cloud mode (optional) | not started |

See `PLAN.md` for the full task list per module.

## Quick start

Requires Python 3.11.

```bash
make setup     # create .venv and install pinned dependencies
make test      # run the test suite
make help      # list every target
```

Local mode is the default and needs no AWS account. Cloud mode is opt-in via `ZTB_MODE=cloud`.

## Repository layout

```
ztb/
  config.py        constants from the paper (α, λ, β, bands, dimensions) — single source of truth
  features/        CERT→CloudTrail mapper, label attachment, baseline store, feature builder
  risk/            autoencoder.py, iforest.py, fusion.py, trust.py
  pdp/             FastAPI app (Algorithm 1), signer, hash chain, evidence store, queue
  ledger/          Fabric client, committer, chain verification
chaincode/         Go chaincode: LogAccess, QueryByPrincipal, QueryByResource, VerifyChain
scripts/           build_dataset, train, evaluate, ablation, latency_bench, tamper_test, demo
infra/             (cloud mode) SAM/Terraform for API Gateway + Lambda + SageMaker + STS
tests/             pytest suite
results/           tables and figures — all generated by scripts, never hand-edited
data/              raw CERT corpus (gitignored) and processed parquet splits
```

## Data

The CERT r4.2 insider-threat corpus is licence-restricted and is **not** committed. Place the
extracted CSVs (`logon.csv`, `file.csv`, `device.csv`, `http.csv`, `LDAP/`, `answers/`) under
`data/raw/r4.2/`, then:

```bash
make sample     # 1-month validation pass -> data/processed/sample/
make dataset    # full 17 months -> data/processed/{train,val,test}.parquet
```

The split is by time — first 10 months train (benign only), next 2 validation, last 5 test —
never random. The full run takes about two hours; on macOS the Makefile wraps it in
`caffeinate -dims` so the machine cannot idle-sleep and eject the external disk mid-run. The mapper streams every file, so `http.csv` (~10 GB) is never loaded into memory;
measured peak RSS is flat at ~135 MB regardless of corpus size.

Budget roughly **3.2 GB** for `data/processed/` on top of the raw corpus.

Full 17-month run (`results/module1_dataset_report.json`, 1 h 24 min on an M2, RSS flat at ~130 MB):

| Split | Rows | Positives | Rate |
|---|---:|---:|---:|
| train (benign only) | 19,209,477 | 0 | — |
| train_malicious (sidelined scenario rows) | 3,975 | 3,975 | — |
| val | 3,251,437 | 1,124 | 0.035 % |
| test | 7,675,354 | 1,754 | 0.023 % |

30,140,243 events in total (94 % http), 1,000 principals; 6,853 of the 7,323 malicious ids in
`answers/` matched — the missing 470 are `email.csv` events, which the paper's mapping does not
cover. The paper's test window (2,184,663 events, 0.179 % positive) is a different cut of the
corpus; ours is larger and sparser because http dominates and most scenarios begin inside the
training window.

`data/raw` is a symlink to an external disk (the corpus is ~20 GB extracted) and is gitignored;
`answers/` ships as a separate archive on KiltHub and must sit at `data/raw/r4.2/answers/`.

Two points about the mapping, both verified against the real r4.2 files and documented in the
source:

- r4.2's `file.csv` has no `activity` column and its filenames are bare (0 of 445,581 rows carry
  a drive letter); every row is a file copied to removable media, so all file events map to
  `s3:GetObject` with the egress marker. r4.2 therefore has no source for `s3:PutObject`; the
  paper's Get/Put split needs the `activity` column of r5.x+ — see `ztb/features/cert_mapper.py`.
- CERT carries no ASN, geolocation, device fingerprint or MFA data, so the five network-and-device
  features are synthesised deterministically from the originating host — see
  `ztb/features/builder.py`.

## Paper → script map

| Paper artefact | Produced by |
|---|---|
| Table V (detection performance) | `scripts/evaluate.py` → `results/table_v.csv` |
| Table VI (ablation and sensitivity) | `scripts/ablation.py` → `results/table_vi.csv` |
| Table VII (ledger cost and tamper) | `scripts/tamper_test.py` → `results/table_vii.csv` |
| Fig. 3, 4 (bar chart, ROC) | `scripts/evaluate.py` → `results/fig3.png`, `results/fig4.png` |
| Fig. 5 (latency by stage) | `scripts/latency_bench.py` → `results/fig5.csv` |
| Fig. 6 (ledger throughput) | `scripts/ledger_bench.py` → `results/fig6.csv` |

## Risk engine (Module 2)

```bash
make train                      # 5 seeds -> models/   (--data / --models / --out on every script)
make eval                       # Table V, Table VI, Fig. 3, Fig. 4 -> results/
make eval DATA=/x OUT=/tmp/o    # any split directory with the build_dataset schema
```

Two places where the implementation departs from the paper's text, both deliberate:

- **eq. (4) as typeset**, `1 − (1−r)^(1+λs)·(1−βc)`, makes the compensating-control credit
  *raise* risk (R = βc for a perfectly normal request). `ztb/risk/trust.py` implements the evident
  intent, `[1 − (1−r)^(1+λs)]·(1−βc)`, and keeps `literal=True` for comparison.
- **The supervised baselines** need malicious training rows, but the training window is benign-only
  by construction. `build_dataset` writes the scripted-scenario rows it removes from the training
  window to `train_malicious.parquet`; the baselines train on benign train + that file.

- **The compensating-control credit c is 0 throughout the CERT replay.** The corpus has no MFA
  or managed-device signal, and deriving c from the synthesised device features caps R at 0.75
  for nearly every request, making DENY unreachable. The PDP computes c from a real
  authentication context in Module 3.

`evaluate.py` reports every model at the paper's operating threshold R ≥ 0.85 **and** at a
threshold tuned for F1 on the validation window, because a benign-quantile r cannot give a 0.4 %
false-positive rate at R ≥ 0.85 (see the Module 2 report).

## Results on the real corpus (Module 2)

Five seeds, 1 M-row training subsample per seed, evaluated on a seeded 4,000,767-row sample of the
test window (947 positives). Supervised baselines trained on benign train + `train_malicious`.
Source: `results/table_v.csv`, `results/table_v_per_seed.json`, `results/fig3.png`, `results/fig4.png`.

| Model | AUC | Prec. / Rec. / F1 @ R ≥ 0.85 | FPR @ R ≥ 0.85 | F1 @ val-tuned threshold |
|---|---:|---:|---:|---:|
| Logistic regression † | 0.940 | 0.000 / 1.000 / 0.001 | 50.8 % | 0.003 |
| Random forest † | **0.997** | 0.077 / 0.665 / 0.139 | 0.19 % | **0.269** |
| Isolation forest | 0.774 | 0.000 / 0.723 / 0.001 | 41.8 % | 0.005 |
| Deep autoencoder | 0.680 | 0.000 / 0.975 / 0.000 | 92.7 % | 0.002 |
| Proposed hybrid (α = 0.6) | 0.748 | 0.000 / 0.921 / 0.001 | 72.0 % | 0.004 |

† supervised. Paper (Table V): hybrid F1 0.927, AUC 0.964, FPR 0.41 %.

**The paper's unsupervised numbers do not reproduce on this replay, and the reason is
identifiable.** `results/module2_auc_by_source.csv` breaks the validation-window AUC down by CERT
source: 94 % of events are http, so the benign calibration CDFs are http-shaped, and *every*
logon/device/file event — structurally different in its action, resource and sensitivity features —
lands at benign r ≈ 0.94 regardless of intent. The engine has learned "is not http", not "is unusual
for this principal". Device events, where 566 of the 1,124 validation positives live, score AUC 0.56;
the corpus-wide 0.82 on val is almost entirely the http subset.

Two consequences. First, the features do carry the signal: the random forest reaches AUC 0.997 on
identical features and splits. It is the single-manifold, globally-calibrated unsupervised design
that fails at event level here. Second, the paper's test window (2,184,663 events) is about a quarter
of ours, which is consistent with http not having been replayed at event level in the paper; that
alone would remove the dominance effect. The cheapest principled fix within the paper's design is to
fit F̂ₑ and F̂ₛ **per event source**, so r reads as "the fraction of benign events *of this kind*
this one exceeds"; it needs no retraining and is left as the next step.

The fixed operating threshold R ≥ 0.85 is unusable for the unsupervised models on any replay where r
is a benign quantile (FPR 42–93 % here); the val-tuned column is the honest operating point.

**Ablation (Table VI, `results/table_vi.csv`).** At the fixed threshold every variant has F1 ≈ 0.001,
so the paper's ΔF1 column carries no information here; the AUC column does, and it is consistent
with the mechanism above:

| Variant | AUC | Paper ΔF1 |
|---|---:|---:|
| Full system (α = 0.6) | 0.748 | baseline |
| Without isolation forest (α = 1.0) | 0.680 | −0.033 |
| Without autoencoder (α = 0.0) | **0.774** | −0.087 |
| Global instead of per-principal action frequency | 0.733 | −0.051 |
| Without sliding-window rate features | 0.764 | −0.024 |
| Without resource sensitivity in eq. (4) | 0.734 | −0.016 |
| α = 0.4 | 0.762 | −0.006 |
| α = 0.8 | 0.723 | −0.009 |

Removing the autoencoder *improves* AUC and shifting α toward the forest helps — the reverse of the
paper, and what an http-shaped reconstruction manifold predicts. The one feature-level result that
survives is the paper's headline one: per-principal action frequency beats a global one.

### Progression: global → per-source calibration → per-source models

`results/table_v_variants.csv` (built by `scripts/compare_variants.py` from `results/global/`,
`results/per_source_calibration/`, `results/per_source_models/`; 5 seeds each, same 4,000,767-row
test sample, 947 positives):

| Model | global AUC | per-source calibration AUC | per-source models AUC |
|---|---:|---:|---:|
| Random forest † | 0.997 | 0.997 | 0.997 |
| Logistic regression † | 0.940 | 0.940 | 0.940 |
| Isolation forest | 0.774 | 0.469 | 0.570 |
| Deep autoencoder | 0.680 | 0.466 | 0.441 |
| Proposed hybrid | **0.748** | **0.481** | **0.525** |

† supervised, unaffected by calibration by construction.

- **Per-source calibration** (one CDF pair per event type — `sts`, `egress`, `http` on r4.2 — fitted
  on the benign validation rows of that type, no retraining) removes the "is not http" artefact
  entirely: benign r is uniform inside every type. What remains is the detectors' within-type
  signal, and on the test window it is **chance** (hybrid 0.48). The global 0.75 was therefore
  mostly a class-prior effect: malicious events are disproportionately non-http, and non-http
  events sat at the top of an http-shaped benign quantile.
- **Per-source models** (one autoencoder + forest per type, same architecture and
  hyperparameters, trained on 43k `sts`, 42k `egress` and 1.4 M `http` benign rows per seed,
  fused per type) recover a little forest signal (0.57) but the autoencoder stays *below* 0.5:
  within their type, malicious rows reconstruct slightly *better* than benign ones. Hybrid 0.53.

**Sanity check (`results/sanity_injection.csv`, `scripts/sanity_inject.py`).** Before concluding
"design problem", the pipeline was checked with synthetic anomalies injected into the validation
window and scored by the trained global engine (seed 0, 20,000 perturbed benign rows vs 200,000
untouched):

| Injection | AE AUC | forest AUC | hybrid AUC | injected rows at r ≥ 0.85 |
|---|---:|---:|---:|---:|
| 3 AM removable-media event at 50× volume | 1.000 | 0.977 | 0.997 | 100 % |
| 50× volume only | 1.000 | 0.955 | 0.992 | 100 % |
| novel host + first-time action | 0.997 | 0.863 | 0.956 | 100 % |
| 3 AM only | 0.807 | 0.792 | 0.813 | 26 % |
| Gaussian noise, 3σ | 1.000 | 0.991 | 0.999 | 100 % |

The detectors catch gross off-manifold behaviour exactly as intended. The scripted CERT scenarios
are not off-manifold at event level in this feature space.

**User-day granularity (`results/userday/`, `scripts/userday.py`).** Each (principal, day)
becomes one 75-dim vector — log event count, after-hours fraction, per-feature mean and max, log
counts of the rare flags — and the same autoencoder + forest is trained on benign user-days. With
the training window on the unplugged disk, the validation window was split by day: its first half
trains (19,522 user-days, 94 positive — labels used only by the supervised rows), its second half
calibrates (16,741); the test window is untouched (83,986 user-days, 240 positive). Five seeds.

| Model | user-day AUC | user-day F1 @ R≥0.85 | user-day tuned F1 |
|---|---:|---:|---:|
| Logistic regression † | **0.954** | 0.102 | 0.077 |
| Random forest † | **0.911** | 0.105 | 0.119 |
| Isolation forest | 0.791 | 0.013 | 0.002 |
| Deep autoencoder | 0.545 | 0.006 | 0.008 |
| Proposed hybrid | 0.800 | 0.006 | 0.005 |

Propagated back to requests as r′ = max(request r, the principal's current-day r), on the same
4 M-row test sample (`table_v_propagated.csv`): request-only AUC 0.748 →
user-day-only 0.788 → propagated **0.799**. Aggregation is where the
unsupervised signal lives: the forest reaches 0.79 on user-days against 0.47–0.57 on events, the
autoencoder stays near chance, and the hybrid ends at 0.80 at both granularities.

**Operating configuration.** Nothing unsupervised reaches AUC 0.85. The configurations that do are
**supervised at user-day granularity**: logistic regression 0.954 and random forest
0.911 on the 75-dim user-day vector. That is the configuration the PDP demo should
use for its risk score, with the caveat that it is trained with scenario labels (from the
validation window's first half here; from `train_malicious` once the training window is back)
and therefore reflects the paper's supervised baselines rather than its unsupervised design.

Conclusion for the write-up: on CERT r4.2 replayed at event level, the paper's unsupervised
design — reconstruction and isolation over a per-request behavioural vector — does not separate
the scripted scenarios from normal activity, in any of the three configurations, while a
supervised model on the identical vectors does (0.997). The information the scenarios leave in
these features is real but is not "off-manifold" in the sense the detectors look for; it is
conditional structure a discriminative model picks up. Detecting it unsupervised would need
different features (session- or day-level aggregates, content/topic signals from the http and
file payloads the mapping discards) rather than a different calibration. The ablation for the
per-source-calibration variant is in `results/per_source_calibration/table_vi.csv`.

## PDP service (Module 3, local mode)

```bash
make serve                                   # ZTB_MODELS=models uvicorn ztb.pdp.app:app --port 8000
make latency                                 # Fig. 5: 50k requests at 200 rps -> results/fig5.csv
curl -s localhost:8000/authorize -H 'content-type: application/json' -d '{
  "principal": "CDE1846", "action": "s3:GetObject",
  "resource": "arn:aws:s3:::ztb-sales/docs/q3.pdf",
  "context": {"device_id": "PC-0001", "device_managed": true,
              "mfa_age_seconds": 60, "mfa_hardware_backed": true}}'
curl -s localhost:8000/records/CDE1846        # this principal's audit records, in seq order
curl -s localhost:8000/verify/CDE1846         # VerifyChain: first discontinuity or intact
```

`POST /authorize` implements Algorithm 1 line by line and returns the verdict, the scoped
credential (signed mock token locally; STS behind `ZTB_MODE=cloud`), the signed audit record of
eq. (5), the risk breakdown and per-stage timings. Static entitlement (`ztb/pdp/policies/*.json`,
RBAC + ABAC, deny-by-default) is evaluated first and a static denial is final. A STEP-UP verdict
returns a signed challenge; presenting it as `step_up_token` on the retry counts as a fresh
hardware-backed MFA on a managed device.

**The compensating-control credit c comes from the request's auth context here** (managed device
× MFA freshness over 8 h × hardware-backed bonus), which is what the CERT replay could not supply.
A stolen key on an unmanaged host gets no credit; a user who just passed hardware MFA on a managed
laptop gets the full β = 0.25 discount.

**Fig. 5 (`results/fig5.csv`, `results/fig5_meta.json`).** 50,000 requests at a steady 200 rps
against the real models, in-process through an ASGI transport, all 50,000 committed by the
asynchronous committer:

| Stage | p50 | p95 |
|---|---:|---:|
| static policy | 0.018 ms | 0.023 ms |
| feature assembly | 0.128 ms | 0.213 ms |
| inference (AE + forest + CDFs) | 2.087 ms | 2.184 ms |
| trust + band | 0.017 ms | 0.019 ms |
| credential | 0.002 ms | 0.092 ms |
| record: digest, sign, enqueue | 0.269 ms | 0.319 ms |
| **total added latency** | **2.573 ms** | **2.785 ms** |

The paper reports 61.3 ms median / 109.8 ms p95, which includes the API Gateway → Lambda →
SageMaker network hops; these figures are the PDP's own cost with the model co-located. Ledger
commitment is off the path in both.

Records are ECDSA-P-256-signed over a canonical JSON of the tuple, hash-chained per principal
(`seq`, `prevHash`), and carry `h_feat = SHA-256(x ‖ salt)`; the vector and salt go to a
Fernet-encrypted off-chain store under `state/evidence/`. The committer drains an asyncio queue
into an append-only JSONL ledger under `state/ledger/` until the Fabric client lands in Module 4.

## Demo dashboard (Module 3b)

```bash
make demo-scenario      # picks the test-window insider with the most scripted events -> demo/
make demo               # streamlit run ztb/demo/app.py
```

Split screen over one CERT insider (`HBO0413`: 4,578 requests over 42 days, 229 scripted-malicious,
removable-media + web), replayed day by day at an adjustable speed.

- **Left, Plain IAM** — entitlement only: every request is ALLOW and the log is a mutable list.
- **Right, ZTBAudit** — the operating configuration from the Progression section: the random
  forest on the principal's user-day vector gives the day's risk (SHAP top-3 contributing features
  shown), propagated to each request as r′ = max(request-level r, day r), through eq. (4) and
  Table III into a live gauge and verdict colour; every request becomes a signed, hash-chained
  record on the reference ledger with `VerifyChain` status.
- **Cover tracks** — A3 deletes three suspicious records and rewrites three verdicts to ALLOW in
  *both* stores. The plain log just gets shorter; the ledger's `VerifyChain` reports the first
  discontinuity and the broken link is highlighted.

With the test-network and shim up the dashboard uses the **live Fabric ledger** (auto-detected;
sidebar shows `ledger: fabric`, committed / pending / rejected): each request's record is signed,
chained and committed by an ordered background thread — the PDP's asynchronous committer — so the
ledger lags the replay by the block cadence; "Cover tracks" then edits the endorsing peer's
CouchDB directly and `VerifyChain` on that peer reports the break. Without the network it falls
back to `ztb/ledger/sim.py`. The peers must run with the CouchDB state cache disabled for
direct edits to be visible immediately (see `chaincode/README.md`). The scenario parquet is
derived from the licence-restricted corpus and is not committed; regenerate it with
`make demo-scenario`.

## Ledger (Module 4)

`chaincode/auditcontract/` (Go) implements `LogAccess`, `QueryByPrincipal`, `QueryByResource`,
`VerifyChain` and a one-time `SetPDPPublicKey`; there is no update or delete. `LogAccess` verifies
the PDP's ECDSA-P-256 signature over a byte-exact port of the PDP's canonical JSON (the Go tests
verify signatures made by `ztb/pdp/signer.py`), enforces per-principal `seq` continuity and
`prevHash`, and rejects duplicate `recId`. `VerifyChain` re-walks the chain and checks the record
count against the stored head, so a deleted *last* record is caught as well as an interior one.

```bash
cd chaincode/auditcontract && go test ./...             # no network needed
ZTB_LEDGER=sim   make serve                              # reference ledger, in-process
ZTB_LEDGER=fabric make serve                             # via ztb/ledger/shim (Node, Fabric Gateway)
python scripts/tamper_test.py  --ledger sim|fabric       # Table VII
python scripts/ledger_bench.py --ledger fabric           # Fig. 6
```

`chaincode/README.md` has the fabric-samples test-network setup (2 orgs, CouchDB), deployment,
the shim, and how to extend to the paper's three organisations.

**Table VII (`results/table_vii.csv`).** 500 tamper attempts by a privileged log manipulator —
125 each of delete, modify-verdict, backdate, fabricate — against a mirrored JSONL log and the
ledger, plus 10,000 untampered control records:

| Metric | JSONL log | Ledger |
|---|---:|---:|
| attempts that succeeded | 500 / 500 | — |
| attempts self-evident from the log alone | 0 / 500 | — |
| tampering attempts detected | 0 / 500 | **500 / 500** |
| false alarms over clean records | — | **0 / 10000** |

**These are live-network numbers** (`ledger=fabric` in the CSV): a two-organisation
fabric-samples test-network with CouchDB, the chaincode deployed as a service, records committed
through the Fabric Gateway. Fabrication is rejected at `LogAccess` (endorsement fails on the
signature); deletion, verdict edits and backdating were applied *directly to the endorsing peer's
CouchDB* — the privileged-manipulator move — and every one is caught by `VerifyChain`, including
deletion of a chain's last record (the head check). The same experiment against the Python
reference ledger, run before the network existed, is kept under `results/sim/` and agrees
(500/500, 0/10,000).

**Fig. 6 (`results/fig6.csv`, live network).** Offered load vs committed throughput and commit
latency, 32 concurrent submitters for 20 s per rate, 200 fresh principals per rate:

| Offered tx/s | Committed tx/s | commit p50 | commit p95 |
|---:|---:|---:|---:|
| 25 | 24.9 | 307.5 ms | 488.3 ms |
| 50 | 49.6 | 245.1 ms | 505.6 ms |
| 100 | 53.8 | 516.4 ms | 753.3 ms |
| 200 | 50.3 | 563.0 ms | 799.9 ms |
| 300 | 50.5 | 549.5 ms | 811.9 ms |

The network tracks offered load to 50 tx/s and saturates at about 50 tx/s (paper: ~450 tx/s on
nine t3.medium hosts with a three-node Raft orderer; this is one laptop running two peers, one
orderer and two CouchDBs under Docker Desktop with default 2 s batch timeout). Ledger commitment
stays off the authorisation path either way.

Deployment notes that cost real time: Fabric 2.5's in-peer chaincode build does not work with
current Docker Desktop engines (empty build log, broken pipe), so the chaincode runs as a service
(`chaincode/README.md`); `contractapi` rejects pointer fields in returned structs and treats
every field as required unless tagged `metadata:",optional"`; sequential Gateway submits commit
at ~0.3 tx/s because each waits for its block, so the experiment commits chains in parallel.

## Notes

- Results in this repo are reproductions on a laptop-scale testbed; the paper's figures come from
  an AWS deployment. Divergences are recorded in each module's report.
- Nothing in `results/` is typed by hand — regenerate it with `make eval`.
