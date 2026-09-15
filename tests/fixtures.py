"""Synthetic CERT r4.2 corpus, written to the documented schema.

The real corpus is licence-restricted and ~20 GB, so the Module 1 tests run against a
miniature one generated here: same file names, same headers, same timestamp format,
same brace-delimited event ids, same LDAP and answers/ layout. It is small enough to
assert on exactly, and it exercises the mapper, the labels, the baseline decay, the
sliding windows and the time split.

This is a test aid, not a substitute for validating against the real r4.2 corpus.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from sentinel.features.schema import CERT_HEADERS, CERT_TIME_FORMAT, LDAP_HEADER

USERS = ["AAM0658", "BCD1234", "CDE1846", "DEF2222"]
PCS = ["PC-0001", "PC-0002", "PC-0003"]
UNITS = ["1 - Research", "2 - Sales", "3 - Engineering", "4 - Engineering"]
START = datetime(2010, 1, 4, 8, 0, 0)


def _event_id(rng: random.Random) -> str:
    def block(n: int) -> str:
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(n))
    return "{" + f"{block(4)}-{block(8)}-{block(8)}" + "}"


def write_corpus(
    root: Path,
    *,
    days: int = 40,
    events_per_day: int = 12,
    seed: int = 7,
) -> dict[str, object]:
    """Write a synthetic r4.2 corpus under ``root`` and return ground truth about it."""
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[list[str]]] = {k: [] for k in ("logon", "device", "file", "http")}
    malicious_ids: list[str] = []
    all_ids: list[str] = []
    insider = "CDE1846"
    insider_events = 0

    for day in range(days):
        for slot in range(events_per_day):
            ts = START + timedelta(days=day, hours=slot // 3, minutes=17 * (slot % 3))
            user = USERS[slot % len(USERS)]
            pc = PCS[slot % len(PCS)]
            eid = _event_id(rng)
            all_ids.append(eid)
            stamp = ts.strftime(CERT_TIME_FORMAT)
            kind = slot % 4
            if kind == 0:
                rows["logon"].append([eid, stamp, user, pc,
                                      "Logon" if slot % 8 == 0 else "Logoff"])
            elif kind == 1:
                rows["device"].append([eid, stamp, user, pc, "Connect"])
            elif kind == 2:
                # Real r4.2 filenames are bare and content holds the file's magic bytes.
                fname = "EYPC9Y08.doc" if slot % 8 == 2 else "N3LTSU3O.pdf"
                magic = "D0-CF-11-E0-A1-B1-1A-E1" if fname.endswith(".doc") else "25-50-44-46-2D"
                rows["file"].append([eid, stamp, user, pc, fname, magic + " lorem ipsum"])
            else:
                rows["http"].append([eid, stamp, user, pc,
                                     "http://wikileaks.org/leak", "page-content-" * 5])

            # Mark a thin slice of late-window events malicious, mimicking a scripted
            # scenario confined to one insider. Keyed off a per-insider counter rather
            # than the slot layout, so the marking survives any events_per_day.
            if day >= days - 10 and user == insider:
                insider_events += 1
                if insider_events % 2 == 0:
                    malicious_ids.append(eid)

    for source, data in rows.items():
        path = root / f"{source}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(CERT_HEADERS[source])
            writer.writerows(data)

    # LDAP monthly snapshots
    ldap = root / "LDAP"
    ldap.mkdir(exist_ok=True)
    for month in ("2010-01", "2010-02"):
        with (ldap / f"{month}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(LDAP_HEADER)
            for i, user in enumerate(USERS):
                writer.writerow([
                    f"Employee {i}", user, f"{user}@dtaa.com",
                    "ITAdmin" if i == 0 else "Salesman",
                    UNITS[i], UNITS[i], f"Dept {i}", f"Team {i}",
                    "" if i == 0 else "Employee 0",
                ])

    # answers/ tree, nested per scenario as in the real release
    answers = root / "answers" / "r4.2-2"
    answers.mkdir(parents=True, exist_ok=True)
    with (answers / "r4.2-2-CDE1846.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)  # real answer files carry no header row
        for eid in malicious_ids:
            writer.writerow(["file", eid, "01/01/2010 00:00:00", "CDE1846", "PC-0001", "x"])
    # insiders.csv in the real release covers every dataset with a bare "4.2" column.
    # A decoy r5.2 row and a decoy r4.1 scenario file prove the release filter works.
    with (root / "answers" / "insiders.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "scenario", "details", "user", "start", "end"])
        writer.writerow(["4.2", "2", "r4.2-2-CDE1846.csv", "CDE1846",
                         "01/01/2010 00:00:00", "12/31/2010 00:00:00"])
        writer.writerow(["5.2", "1", "r5.2-1-ZZZ9999.csv", "ZZZ9999",
                         "01/01/2010 00:00:00", "12/31/2010 00:00:00"])
    with (root / "answers" / "r4.1-1.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["logon", _event_id(rng), "01/01/2010 00:00:00", "ZZZ9999",
                         "PC-0000", "Logon"])

    return {
        "root": root,
        "total_events": len(all_ids),
        "malicious_ids": set(malicious_ids),
        "users": set(USERS),
        "rows_per_source": {k: len(v) for k, v in rows.items()},
        "start": START,
        "days": days,
    }


# --------------------------------------------------------------------------
# Synthetic 34-dim dataset for the risk engine (Module 2)
# --------------------------------------------------------------------------
# Same parquet schema as scripts/build_dataset.py, so every Module 2 script can be
# pointed at either this or data/processed/ with --data. Benign rows lie on a
# low-dimensional nonlinear manifold with correlated features; anomalies are either
# off-manifold noise or on-manifold points with a joint deviation across correlated
# features (the case the paper says the autoencoder catches and the forest misses).

import json  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sentinel.features.builder import FEATURE_NAMES, StandardiserAccumulator  # noqa: E402

LATENT_DIM = 5


def _benign_manifold(rng: np.random.Generator, n: int, proj: np.ndarray,
                     bias: np.ndarray) -> np.ndarray:
    """Benign feature matrix (n, 34): nonlinear image of a 5-dim latent plus noise."""
    z = rng.normal(size=(n, LATENT_DIM))
    x = np.tanh(z @ proj + bias) * 2.0 + rng.normal(scale=0.08, size=(n, len(FEATURE_NAMES)))
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # Make the named features look like the real ones so trust/ablation code paths
    # that read them by name (sensitivity, credit, rate features) are exercised.
    angle = (z[:, 0] * 0.9) % (2 * np.pi)
    x[:, idx["hour_sin"]] = np.sin(angle)
    x[:, idx["hour_cos"]] = np.cos(angle)
    dow = (z[:, 1] * 0.7) % (2 * np.pi)
    x[:, idx["dow_sin"]] = np.sin(dow)
    x[:, idx["dow_cos"]] = np.cos(dow)
    for flag in ("first_time_action", "first_time_resource", "asn_novelty", "host_novelty",
                 "cross_account_flag", "is_supervisor", "principal_type"):
        x[:, idx[flag]] = (z[:, 2] > 2.2).astype(float)          # rare, correlated
    x[:, idx["is_session_action"]] = (z[:, 3] > 1.0).astype(float)
    x[:, idx["is_egress_action"]] = 1.0 - x[:, idx["is_session_action"]]
    x[:, idx["device_fingerprint_match"]] = (z[:, 2] <= 2.2).astype(float)
    x[:, idx["resource_sensitivity"]] = np.select(
        [z[:, 4] < -0.5, z[:, 4] < 0.5, z[:, 4] < 1.2], [0.3, 0.6, 0.8], 1.0)
    x[:, idx["action_frequency"]] = np.clip(0.5 + 0.25 * np.tanh(z[:, 3]), 0.0, 1.0)
    x[:, idx["read_write_ratio"]] = np.clip(0.9 + 0.05 * z[:, 1], 0.0, 1.0)
    # Rate features: a consistent burst structure (1m <= 15m <= 60m).
    base = np.exp(1.0 + 0.6 * z[:, 0])
    x[:, idx["calls_1min"]] = np.floor(base * 0.1)
    x[:, idx["calls_15min"]] = np.floor(base * 0.5)
    x[:, idx["calls_60min"]] = np.floor(base)
    x[:, idx["distinct_resources_60min"]] = np.floor(base * 0.6)
    x[:, idx["distinct_actions_60min"]] = np.clip(np.floor(base * 0.1), 1, 4)
    x[:, idx["log_bytes_read_60min"]] = np.log1p(base * 300)
    x[:, idx["log_mfa_age"]] = np.log1p(np.exp(7.5 + 0.8 * z[:, 1]))
    x[:, idx["geodesic_distance_km"]] = 0.0
    return x


def _anomalies(rng: np.random.Generator, benign: np.ndarray, n: int) -> np.ndarray:
    """Half off-manifold, half joint-deviation anomalies drawn from benign rows."""
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    src = benign[rng.integers(0, len(benign), size=n)].copy()
    half = n // 2

    # (a) off-manifold: large shifts on several random continuous dims
    cont = [idx[f] for f in ("role_tenure_days", "policy_breadth", "log_inter_request_gap",
                             "action_class_entropy", "log_action_history",
                             "arn_prefix_depth", "log_bytes_read_60min", "log_mfa_age")]
    for row in src[:half]:
        dims = rng.choice(cont, size=4, replace=False)
        row[dims] += rng.choice([-1, 1], size=4) * rng.uniform(3.0, 6.0, size=4)
        row[idx["geodesic_distance_km"]] = rng.uniform(2000, 12000)
        row[idx["asn_novelty"]] = 1.0
        row[idx["host_novelty"]] = 1.0

    # (b) joint deviation: individually plausible values in an impossible combination
    for row in src[half:]:
        row[idx["calls_1min"]] = row[idx["calls_60min"]] * 3 + 40      # burst > hour total
        row[idx["calls_15min"]] = row[idx["calls_1min"]] * 0.5
        row[idx["first_time_action"]] = 1.0
        row[idx["action_frequency"]] = 0.98                              # "novel" yet frequent
        row[idx["is_egress_action"]] = 1.0
        row[idx["resource_sensitivity"]] = 1.0
        row[idx["device_fingerprint_match"]] = 0.0
    return src


def write_synthetic_splits(
    out_dir: Path,
    *,
    n_train: int = 20_000,
    n_val: int = 5_000,
    n_test: int = 10_000,
    anomaly_rate: float = 0.02,
    n_train_malicious: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Write train/train_malicious/val/test parquet + standardiser.json under ``out_dir``.

    Returns ground truth (positives per split) for assertions.
    """
    from build_dataset import arrow_schema  # scripts/ is on sys.path in tests

    rng = np.random.default_rng(seed)
    proj = rng.normal(scale=0.8, size=(LATENT_DIM, len(FEATURE_NAMES)))
    bias = rng.normal(scale=0.3, size=len(FEATURE_NAMES))
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = arrow_schema()
    t0 = datetime(2010, 1, 1)
    truth: dict[str, object] = {"out_dir": out_dir}
    counter = 0

    def emit(name: str, benign_n: int, anomalous_n: int, day0: int) -> None:
        nonlocal counter
        b = _benign_manifold(rng, benign_n, proj, bias)
        rows = [b]
        labels = [np.zeros(benign_n, dtype=np.int8)]
        if anomalous_n:
            # Anomalies are perturbed benign rows; an all-malicious split (train_malicious)
            # needs a benign pool to perturb even though none of it is written.
            pool = b if benign_n else _benign_manifold(rng, 1000, proj, bias)
            rows.append(_anomalies(rng, pool, anomalous_n))
            labels.append(np.ones(anomalous_n, dtype=np.int8))
        x = np.vstack(rows)
        y = np.concatenate(labels)
        order = rng.permutation(len(x))
        x, y = x[order], y[order]
        n = len(x)
        table = {
            "event_id": [f"{{SYN{counter + i:012d}}}" for i in range(n)],
            "ts": [t0 + timedelta(days=day0, seconds=int(i * 86400 * 30 / n)) for i in range(n)],
            "principal": [f"U{rng.integers(0, 200):04d}" for _ in range(n)],
            "action": rng.choice(["s3:GetObject", "execute-api:Invoke", "sts:AssumeRole",
                                  "sts:SessionEnd"], size=n, p=[0.05, 0.9, 0.03, 0.02]).tolist(),
            "resource": ["arn:aws:s3:::sentinel-synth/x"] * n,
            "source": ["synthetic"] * n,
            "label": y.tolist(),
        }
        counter += n
        for i, fname in enumerate(FEATURE_NAMES):
            table[fname] = x[:, i].astype(np.float64).tolist()
        pq.write_table(pa.Table.from_pydict(table, schema=schema), out_dir / f"{name}.parquet")
        truth[name] = {"rows": n, "positives": int(y.sum())}
        if name == "train":
            acc = StandardiserAccumulator()
            for row in x:
                acc.update(row)
            (out_dir / "standardiser.json").write_text(json.dumps(acc.finalize().to_dict()))

    emit("train", n_train, 0, day0=0)
    emit("train_malicious", 0, n_train_malicious, day0=100)
    emit("val", n_val, int(n_val * anomaly_rate), day0=305)
    emit("test", n_test, int(n_test * anomaly_rate), day0=366)
    return truth
