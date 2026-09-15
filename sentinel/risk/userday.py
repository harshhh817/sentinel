"""User-day aggregation: one vector per (principal, calendar day).

Most CERT insider-threat work scores principal-days, not events. This layer folds a
split's 34-dim request vectors into per-user-day aggregates -- event count, per-feature
mean and max, after-hours fraction, and counts of the rare flags -- so the same
autoencoder + isolation forest can be trained on benign user-days and score each day.
A user-day is malicious if any of its events is.

Aggregation streams by parquet row group with a running accumulator keyed on
(principal, day); memory is bounded by the number of user-days, not events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sentinel.features.builder import FEATURE_NAMES, Standardiser

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_FLAG_COUNTS = ("is_egress_action", "is_session_action", "first_time_action",
                "first_time_resource", "host_novelty")

USERDAY_FEATURES: tuple[str, ...] = (
    ("log_n_events", "after_hours_frac")
    + tuple(f"mean_{f}" for f in FEATURE_NAMES)
    + tuple(f"max_{f}" for f in FEATURE_NAMES)
    + tuple(f"log_count_{f}" for f in _FLAG_COUNTS)
)
N_USERDAY = len(USERDAY_FEATURES)


def _hours(x: np.ndarray) -> np.ndarray:
    h = np.arctan2(x[:, _IDX["hour_sin"]], x[:, _IDX["hour_cos"]]) / (2 * np.pi) * 24.0
    return np.mod(h, 24.0)


@dataclass
class UserDays:
    principal: np.ndarray
    day: np.ndarray
    x: np.ndarray           # (n, N_USERDAY) float32
    y: np.ndarray           # (n,) int8: 1 if any event that day is malicious
    n_events: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, m: np.ndarray) -> UserDays:
        return UserDays(self.principal[m], self.day[m], self.x[m], self.y[m], self.n_events[m])

    @property
    def benign(self) -> UserDays:
        return self.subset(self.y == 0)

    def key_to_index(self) -> dict[tuple[str, str], int]:
        return {(p, d): i for i, (p, d) in enumerate(zip(self.principal, self.day, strict=True))}


def aggregate(path: Path, *, progress: bool = False) -> UserDays:
    """Stream a split's parquet into user-day vectors."""
    pf = pq.ParquetFile(path)
    cols = ["principal", "ts", "label"] + list(FEATURE_NAMES)
    # key -> [n, sum(34), max(34), after_hours, flags(5), label]
    acc: dict[tuple[str, str], list] = {}
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        x = np.column_stack([t.column(f).to_numpy(zero_copy_only=False)
                             for f in FEATURE_NAMES]).astype(np.float32)
        prin = t.column("principal").to_numpy(zero_copy_only=False).astype(str)
        day = t.column("ts").to_numpy(zero_copy_only=False).astype("datetime64[D]").astype(str)
        lab = t.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
        hrs = _hours(x)
        after = ((hrs < 6.0) | (hrs >= 22.0)).astype(np.float32)
        flags = x[:, [_IDX[f] for f in _FLAG_COUNTS]]
        df = pd.DataFrame({"p": prin, "d": day})
        for key, idx in df.groupby(["p", "d"], sort=False).indices.items():
            xs = x[idx]
            a = acc.get(key)
            if a is None:
                acc[key] = [len(idx), xs.sum(0), xs.max(0), after[idx].sum(), flags[idx].sum(0),
                            int(lab[idx].max())]
            else:
                a[0] += len(idx)
                a[1] += xs.sum(0)
                np.maximum(a[2], xs.max(0), out=a[2])
                a[3] += after[idx].sum()
                a[4] += flags[idx].sum(0)
                a[5] = max(a[5], int(lab[idx].max()))
        if progress and rg % 200 == 0:
            print(f"    row group {rg}/{pf.num_row_groups}  user-days so far {len(acc):,}",
                  flush=True)
    keys = list(acc)
    n = np.array([acc[k][0] for k in keys], dtype=np.float32)
    x = np.column_stack([
        np.log1p(n),
        np.array([acc[k][3] for k in keys], dtype=np.float32) / n,
        np.vstack([acc[k][1] for k in keys]) / n[:, None],
        np.vstack([acc[k][2] for k in keys]),
        np.log1p(np.vstack([acc[k][4] for k in keys])),
    ]).astype(np.float32)
    y = np.array([acc[k][5] for k in keys], dtype=np.int8)
    return UserDays(np.array([k[0] for k in keys]), np.array([k[1] for k in keys]), x, y, n)


def save(ud: UserDays, path: Path) -> None:
    df = pd.DataFrame(ud.x, columns=list(USERDAY_FEATURES))
    df.insert(0, "n_events", ud.n_events)
    df.insert(0, "label", ud.y)
    df.insert(0, "day", ud.day)
    df.insert(0, "principal", ud.principal)
    df.to_parquet(path, index=False)


def load(path: Path) -> UserDays:
    df = pd.read_parquet(path)
    return UserDays(df["principal"].to_numpy().astype(str), df["day"].to_numpy().astype(str),
                    df[list(USERDAY_FEATURES)].to_numpy(np.float32),
                    df["label"].to_numpy(np.int8), df["n_events"].to_numpy(np.float32))


def fit_standardiser(x: np.ndarray) -> Standardiser:
    """All user-day features are continuous; standardise every column."""
    mean = x.astype(np.float64).mean(0)
    std = x.astype(np.float64).std(0)
    std[std < 1e-8] = 1.0
    return Standardiser(mean=mean, std=std, mask=np.ones(x.shape[1], dtype=bool))
