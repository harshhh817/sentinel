# Viva — likely questions and crisp answers

Numbers are from `results/`; every one is produced by a script.

**1. What does ZTBAudit add over IAM?**
IAM answers "is this principal entitled?" once, from a static policy. ZTBAudit re-evaluates every
request against the principal's own behavioural baseline, issues short-lived least-privilege
credentials sized to the risk, and binds the score, feature digest and verdict into a signed,
hash-chained record on a ledger no single administrator controls. Replayed CERT scenarios are
performed by *entitled* principals, so IAM denies 0 of them; that is the gap.

**2. Did you reproduce the paper's detection numbers?**
No. The paper reports hybrid AUC 0.964 / F1 0.927; we get AUC 0.748 and F1 ≈ 0.001 at its
operating threshold. The gap is explained, not hand-waved: see Q3–Q5.

**3. Why did the unsupervised design fail?**
94 % of replayed events are http. The benign calibration CDFs are therefore http-shaped, and every
logon/device/file event — structurally different in action, resource and sensitivity features —
sits at benign quantile ≈ 0.94 regardless of intent. The engine learned "is not http". Per-source
calibration removes that artefact and shows chance-level within-type signal (AUC 0.48); per-source
models reach 0.53. The scripted scenarios are simply not off-manifold at event level in this
feature space.

**4. How do you know it is the data and not a bug?**
Three checks. Injected gross anomalies (3 AM removable-media events at 50× volume) are caught with
AUC 0.997 and 100 % above R ≥ 0.85. A supervised random forest on the *identical* vectors and
splits reaches AUC 0.997, so the features carry the signal. And we found and fixed one real defect
on the way (a test overwrote the standardiser with synthetic statistics), then re-ran.

**5. Why is the random forest the operating model? Isn't that abandoning the paper's idea?**
It is the only configuration above AUC 0.85: 0.954 (LR) / 0.911 (RF) on user-day vectors. The
paper itself lists these as supervised baselines. We kept the paper's architecture — trust
algorithm, bands, credentials, evidence chain — and swapped the risk source, which the design
allows: the PDP consumes r, it does not care how r was produced. The honest caveat: it needs
scenario labels; an unsupervised detector would need session/day aggregates and content features
the CERT→CloudTrail mapping discards.

**6. What is the user-day layer and why does it help?**
One 75-dim vector per (principal, day): event count, after-hours fraction, per-feature mean and
max, counts of rare flags. Most CERT work scores principal-days. The isolation forest goes from
0.47 on events to 0.79 on user-days; propagated back as r′ = max(request r, day r) the request-level
AUC moves 0.748 → 0.799. Aggregation is where the unsupervised signal lives.

**7. What is wrong with equation (4) as printed?**
`R = 1 − (1−r)^(1+λs)·(1−βc)` gives R = βc = 0.25 for a perfectly normal request under full
credit — a "compensating-control credit" that raises risk. That contradicts the paper's own stated
property (R = 0 when r = 0). We implement `[1 − (1−r)^(1+λs)]·(1−βc)`: monotone in r and s,
non-increasing in c, zero at r = 0. `literal=True` reproduces the typeset form for comparison.

**8. Why is the fixed threshold R ≥ 0.85 unusable?**
r is a benign quantile by construction, so for a typical sensitivity roughly a third of benign
traffic exceeds R ≥ 0.85 (we measure 42–93 % FPR across detectors). No quantile score can give the
paper's 0.41 %. We report the paper's operating point *and* a threshold tuned for F1 on the
validation window.

**9. Where does the MFA credit c come from?**
CERT has no MFA or managed-device signal, so c = 0 throughout the replay; deriving it from the
synthesised device features capped R at 0.75 and made DENY unreachable. The PDP computes c from
the request's real authentication context: managed device × MFA freshness (8 h) × hardware-backed
bonus. A stolen key on an unmanaged host gets no credit.

**10. Explain the audit record and what each tamper does to it.**
⟨recId, prevHash, seq, ts, principal, action, resource, r, R, verdict, h_feat, sig⟩. seq is a
per-principal counter, prevHash the SHA-256 of the previous record, h_feat = SHA-256(x ‖ salt) with
x and salt off-chain, sig an ECDSA P-256 signature over the canonical tuple. Modify or backdate →
signature fails. Fabricate → wrong key, rejected at endorsement. Delete → gap in seq/prevHash;
delete the *last* record → caught by the stored head's sequence number (a gap we found and closed).
Live Fabric result: 500/500 detected, 0/10,000 false alarms.

**11. What is the CouchDB cache finding?**
The peer serves recently written keys from an in-memory cache in front of CouchDB, so a direct
CouchDB edit is invisible to `VerifyChain` on that peer until eviction or restart. Our tamper
experiment only saw its edits because 10,000 control writes had churned the cache; the demo runs
peers with the cache disabled. The real lesson: an auditor verifies on a peer *they* control,
whose cache and state the adversary never touched — which is exactly why the ledger spans
organisations.

**12. Why chaincode-as-a-service, and why CouchDB?**
Fabric 2.5's in-peer chaincode build talks a legacy Docker API that current Docker Desktop engines
reject (empty build log); building the image on the host and letting the peer connect avoids it.
CouchDB is needed for `QueryByResource`, a rich query; `QueryByPrincipal` and `VerifyChain` use a
composite key and work on LevelDB.

**13. Is ledger throughput a bottleneck?**
On the laptop network it saturates at ~50 tx/s with ~0.5 s commit latency (paper: ~450 tx/s on
nine EC2 hosts). It never touches the authorisation path: the PDP's added latency is 2.6 ms median
in-process. Per-principal chains serialise at block cadence because each record reads the head
the previous one wrote; batching records per block or anchoring Merkle roots is the scale-out path.

**14. What did the sanity of the data pipeline cost you and what would you do differently?**
The dataset build streams 30 M events at ~135 MB RSS, but the first full attempts died to a
laptop idle-sleep ejecting the external disk, then a full boot volume. Durable fixes: `caffeinate`
around long targets, models and staging on the boot volume. Differently: user-day features from
the start, and a labelled hold-out of scenarios *by scenario id* rather than by time only.

**15. What would a real deployment change?**
Real CloudTrail (the mapping is the main threat to validity); network telemetry so the five
network/device features are measured, not synthesised; c from an IdP's MFA claims; per-organisation
peers with the auditor's own org; batched commits; fail-static policy recorded on the ledger when
the PDP is down; and a supervised or session-level detector retrained on the tenant's own labelled
incidents rather than the paper's unsupervised hybrid.
