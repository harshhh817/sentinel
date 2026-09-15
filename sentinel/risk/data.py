"""Load the parquet splits written by ``scripts/build_dataset.py`` for the risk engine.

Everything here streams by parquet row group so the 19-million-row training window
never has to be materialised. Subsampling is seeded and applied per row group, which
keeps memory at ``max_rows x 34 x 4`` bytes whatever the split size.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sentinel.features.builder import FEATURE_NAMES, Standardiser

META_COLUMNS = ("event_id", "ts", "principal", "action", "resource", "source", "label")


@dataclass
class Split:
    """A feature matrix plus the columns the trust algorithm and metrics need."""

    x: np.ndarray            # (n, d) float32, raw (unstandardised) features
    y: np.ndarray            # (n,) int8 labels
    sensitivity: np.ndarray  # (n,) s in eq. (4)
    credit: np.ndarray       # (n,) c in eq. (4)
    action: np.ndarray | None = None
    source: np.ndarray | None = None
    types: np.ndarray | None = None   # event type per row, see sentinel.risk.types
    principal: np.ndarray | None = None
    day: np.ndarray | None = None     # calendar day as 'YYYY-MM-DD'

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, m: np.ndarray) -> Split:
        pick = lambda a: None if a is None else a[m]  # noqa: E731
        return Split(self.x[m], self.y[m], self.sensitivity[m], self.credit[m],
                     pick(self.action), pick(self.source), pick(self.types),
                     pick(self.principal), pick(self.day))

    @property
    def benign(self) -> Split:
        return self.subset(self.y == 0)


def feature_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def control_credit_from_features(x: np.ndarray) -> np.ndarray:
    """c in eq. (4) for the CERT replay: **zero for every request**.

    The credit is meant for "a recent hardware-backed MFA assertion on a managed
    device". CERT r4.2 carries no MFA, device-management or hardware-attestation
    signal; the ``device_fingerprint_match`` and ``log_mfa_age`` features are
    synthesised from the originating PC and the last logon, which is not evidence of a
    compensating control. Granting credit from them caps R at 1 - beta = 0.75 for
    almost every request and makes DENY unreachable, so the honest replay is c = 0.

    The PDP (Module 3) computes c from a real authentication context instead.
    """
    return np.zeros(len(x), dtype=np.float32)


def num_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def iter_row_groups(
    path: Path,
    *,
    columns: tuple[str, ...] = FEATURE_NAMES,
    with_meta: tuple[str, ...] = ("label",),
) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Yield ``(features, meta)`` per row group; features are float32."""
    pf = pq.ParquetFile(path)
    cols = list(columns) + [c for c in with_meta if c not in columns]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        x = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in columns])
        meta = {c: t.column(c).to_numpy(zero_copy_only=False) for c in with_meta}
        yield x.astype(np.float32, copy=False), meta


def load_split(
    path: Path,
    *,
    max_rows: int | None = None,
    seed: int = 0,
    columns: tuple[str, ...] = FEATURE_NAMES,
    with_action: bool = False,
    with_types: bool = False,
    with_keys: bool = False,
) -> Split:
    """Load a split, optionally as a seeded uniform subsample of ``max_rows`` rows.

    ``with_types`` also reads ``action`` and ``source`` and derives the event type per
    row (needed for per-source calibration and per-source models).
    """
    from sentinel.risk.types import event_type

    total = num_rows(path)
    keep = 1.0 if not max_rows or max_rows >= total else max_rows / total
    rng = np.random.default_rng(seed)
    xs, ys, acts, srcs, prins, days = [], [], [], [], [], []
    meta_cols: tuple[str, ...] = ("label",)
    if with_action or with_types:
        meta_cols += ("action",)
    if with_types:
        meta_cols += ("source",)
    if with_keys:
        meta_cols += ("principal", "ts")
    for x, meta in iter_row_groups(path, columns=columns, with_meta=meta_cols):
        if keep < 1.0:
            m = rng.random(len(x)) < keep
            x = x[m]
            meta = {k: v[m] for k, v in meta.items()}
        xs.append(x)
        ys.append(meta["label"].astype(np.int8))
        if "action" in meta:
            acts.append(meta["action"].astype(str))
        if "source" in meta:
            srcs.append(meta["source"].astype(str))
        if "principal" in meta:
            prins.append(meta["principal"].astype(str))
            days.append(meta["ts"].astype("datetime64[D]").astype(str))
    x = np.vstack(xs) if xs else np.empty((0, len(columns)), np.float32)
    y = np.concatenate(ys) if ys else np.empty(0, np.int8)
    full = x if columns == FEATURE_NAMES else None
    sens = full[:, feature_index("resource_sensitivity")] if full is not None else np.zeros(len(y))
    cred = control_credit_from_features(full) if full is not None else np.zeros(len(y))
    action = np.concatenate(acts) if acts else None
    source = np.concatenate(srcs) if srcs else None
    types = event_type(action, source) if with_types and action is not None else None
    principal = np.concatenate(prins) if prins else None
    day = np.concatenate(days) if days else None
    return Split(x, y, sens.astype(np.float32), cred.astype(np.float32), action, source, types,
                 principal, day)


def load_standardiser(data_dir: Path, models_dir: Path | None = None) -> Standardiser | None:
    """``standardiser.json`` is written next to the parquet files by build_dataset."""
    for d in (data_dir, models_dir):
        if d is not None and (d / "standardiser.json").exists():
            return Standardiser.from_dict(json.loads((d / "standardiser.json").read_text()))
    return None
