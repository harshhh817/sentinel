"""Map the CERT r4.2 corpus onto CloudTrail-style events, streaming.

The corpus does not fit in memory (``http.csv`` alone is ~10 GB), so every reader here
is a generator over ``csv.reader`` and the four sources are combined with a heap-based
k-way merge. Peak memory is the merge heap plus a bounded reorder buffer, both O(1) in
corpus size.

Mapping (Section V of the paper):

===============================  ==========================================
CERT event                       cloud action
===============================  ==========================================
logon.csv  Logon                 ``sts:AssumeRole``
logon.csv  Logoff                ``sts:SessionEnd``
device.csv Connect/Disconnect    ``s3:GetObject`` with an egress marker
file.csv   (every row)           ``s3:GetObject`` with an egress marker
http.csv                         ``execute-api:Invoke`` (external proxy)
===============================  ==========================================

Buckets are assigned from the principal's organisational unit, so the resource string
carries the same org structure the corpus does.

.. note::
   **Verified against the real r4.2 corpus.** ``file.csv`` has no ``activity``
   column, filenames are bare (``EYPC9Y08.doc``; 0 of 445,581 rows carry a drive
   letter), the ``content`` column holds the file's magic bytes, and only 264 of the
   ~1,000 users appear in it. That is the r4.2 semantics: ``file.csv`` records
   files copied **to removable media**, so every row is a read leaving the
   organisation and maps to ``s3:GetObject`` with the egress marker, exactly as the
   paper maps removable-device events. Consequently r4.2 offers no source for
   ``s3:PutObject``; the paper's Get/Put split needs the ``activity`` column of
   r5.x+. The read/write-ratio feature is therefore weak on this corpus (only
   ``sts:SessionEnd`` counts as a write) and is reported as such.
"""

from __future__ import annotations

import csv
import heapq
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from ztb.features.schema import (
    LDAP_HEADER,
    MAPPED_SOURCES,
    CloudEvent,
    SchemaError,
    parse_cert_time,
    validate_header,
)

# CERT's http.csv content field can exceed the default 128 KB csv field limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ACCOUNT = "000000000000"          # single-tenant prototype; the paper uses one account

# Events arriving out of order by more than this are dropped from the merge's
# monotonicity guarantee. CERT files are generated chronologically; the buffer only
# absorbs local jitter within a file.
REORDER_BUFFER = 10_000


# --- LDAP: principal -> organisational unit -------------------------------


def load_org_units(ldap_dir: Path) -> dict[str, dict[str, str]]:
    """Read the monthly LDAP snapshots into ``{user_id: {role, ou, team, supervisor}}``.

    Later snapshots overwrite earlier ones, so the result reflects each user's most
    recent known role. LDAP is small (~1000 users x 17 months) and is loaded eagerly.
    """
    profiles: dict[str, dict[str, str]] = {}
    if not ldap_dir.is_dir():
        return profiles

    for snapshot in sorted(ldap_dir.glob("*.csv")):
        with snapshot.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None:
                continue
            cols = [h.strip() for h in header]
            if tuple(cols) != LDAP_HEADER:
                raise SchemaError(
                    f"{snapshot.name} header mismatch\n"
                    f"  expected: {LDAP_HEADER}\n  actual:   {tuple(cols)}"
                )
            idx = {name: i for i, name in enumerate(cols)}
            for row in reader:
                if len(row) != len(cols):
                    continue
                profiles[row[idx["user_id"]]] = {
                    "role": row[idx["role"]],
                    "business_unit": row[idx["business_unit"]],
                    "functional_unit": row[idx["functional_unit"]],
                    "department": row[idx["department"]],
                    "team": row[idx["team"]],
                    "supervisor": row[idx["supervisor"]],
                    "first_seen": snapshot.stem,
                }
    return profiles


def _slug(value: str) -> str:
    """Make an org-unit string safe for an S3 bucket name."""
    out = "".join(c.lower() if c.isalnum() else "-" for c in value.strip())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "unassigned"


def bucket_for(principal: str, org: dict[str, dict[str, str]]) -> str:
    """The S3 bucket assigned to the principal's organisational unit."""
    profile = org.get(principal)
    unit = profile.get("functional_unit") or profile.get("business_unit") if profile else ""
    return f"ztb-{_slug(unit or 'unassigned')}"


# --- per-source row mapping ------------------------------------------------


def _arn(bucket: str, key: str) -> str:
    return f"arn:aws:s3:::{bucket}/{key}"


def _map_logon(row: dict[str, str], ts: datetime, bucket: str) -> CloudEvent:
    activity = row["activity"].strip().lower()
    action = "sts:AssumeRole" if activity == "logon" else "sts:SessionEnd"
    return CloudEvent(
        event_id=row["id"],
        ts=ts,
        principal=row["user"],
        action=action,
        resource=f"arn:aws:iam::{ACCOUNT}:role/{_slug(bucket)}-session",
        source="logon",
        pc=row["pc"],
        extra={"activity": row["activity"]},
    )


def _map_device(row: dict[str, str], ts: datetime, bucket: str) -> CloudEvent:
    """Removable-media use: a read that leaves the organisation."""
    return CloudEvent(
        event_id=row["id"],
        ts=ts,
        principal=row["user"],
        action="s3:GetObject",
        resource=_arn(bucket, f"removable/{row['pc']}"),
        source="device",
        pc=row["pc"],
        egress=True,
        extra={"activity": row["activity"]},
    )


def _map_file(row: dict[str, str], ts: datetime, bucket: str) -> CloudEvent:
    """A file copied to removable media: a read that leaves the organisation.

    See the module note: on r4.2 every ``file.csv`` row is such a copy.
    """
    filename = row["filename"].strip()
    key = filename.replace("\\", "/").lstrip("/")
    return CloudEvent(
        event_id=row["id"],
        ts=ts,
        principal=row["user"],
        action="s3:GetObject",
        resource=_arn(bucket, f"removable/{key}"),
        source="file",
        pc=row["pc"],
        egress=True,
        # CERT records the file's leading bytes, not a transfer size; the content
        # length is the only volume proxy the corpus offers.
        bytes_read=len(row.get("content", "")),
        extra={"filename": filename},
    )


def _map_http(row: dict[str, str], ts: datetime, bucket: str) -> CloudEvent:
    url = row["url"].strip()
    host = url.split("/")[2] if "://" in url and len(url.split("/")) > 2 else url[:64]
    return CloudEvent(
        event_id=row["id"],
        ts=ts,
        principal=row["user"],
        action="execute-api:Invoke",
        resource=f"arn:aws:execute-api:::{_slug(host)}",
        source="http",
        pc=row["pc"],
        egress=True,
        bytes_read=len(row.get("content", "")),
        extra={"host": host},
    )


_MAPPERS = {
    "logon": _map_logon,
    "device": _map_device,
    "file": _map_file,
    "http": _map_http,
}


# --- streaming readers -----------------------------------------------------


def stream_source(
    path: Path,
    source: str,
    org: dict[str, dict[str, str]],
    *,
    until: datetime | None = None,
) -> Iterator[CloudEvent]:
    """Yield mapped events from one CERT CSV, one row at a time.

    ``until`` stops the scan early once timestamps pass the cutoff, which is what makes
    a one-month sample cheap on a 10 GB file. It assumes the file is broadly
    chronological and allows a slack window before giving up.
    """
    mapper = _MAPPERS[source]
    slack = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return
        validate_header(source, header)
        cols = [h.strip() for h in header]
        for values in reader:
            if len(values) != len(cols):
                continue
            row = dict(zip(cols, values, strict=True))
            try:
                ts = parse_cert_time(row["date"])
            except ValueError:
                continue
            if until is not None and ts > until:
                # Tolerate local disorder before concluding the cutoff is passed.
                slack += 1
                if slack > REORDER_BUFFER:
                    return
                continue
            slack = 0
            yield mapper(row, ts, bucket_for(row["user"], org))


def stream_events(
    root: Path,
    *,
    sources: tuple[str, ...] = MAPPED_SOURCES,
    until: datetime | None = None,
    labels: set[tuple[str, str]] | None = None,
) -> Iterator[CloudEvent]:
    """Merge every CERT source into one stream ordered by timestamp.

    Uses :func:`heapq.merge`, so memory is O(number of sources) regardless of how large
    the underlying files are. ``labels`` is the set of malicious ``(source, event_id)``
    keys from ``answers/``; when supplied, ``CloudEvent.label`` is set as events pass.
    """
    org = load_org_units(root / "LDAP")
    streams = []
    for source in sources:
        path = root / f"{source}.csv"
        if not path.exists():
            continue
        streams.append(stream_source(path, source, org, until=until))

    if not streams:
        raise FileNotFoundError(
            f"no CERT activity files found under {root}. Expected one or more of: "
            + ", ".join(f"{s}.csv" for s in sources)
        )

    merged = heapq.merge(*streams, key=CloudEvent.sort_key)
    if labels is None:
        yield from merged
    else:
        for event in merged:
            event.label = 1 if (event.source, event.event_id) in labels else 0
            yield event
