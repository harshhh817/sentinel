"""CERT r4.2 on-disk schema, and the CloudTrail-style event we map it onto.

The r4.2 release ships one CSV per activity type plus monthly LDAP snapshots and an
``answers/`` tree marking the scripted malicious events. Column orders below are the
documented r4.2 layout; :func:`validate_header` fails loudly with the offending file
and the actual header if a corpus differs, rather than silently mis-parsing 10 GB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# --- CERT r4.2 CSV headers -------------------------------------------------
# Keys are the file stem under data/raw/r4.2/.

CERT_HEADERS: dict[str, tuple[str, ...]] = {
    "logon": ("id", "date", "user", "pc", "activity"),
    "device": ("id", "date", "user", "pc", "activity"),
    "file": ("id", "date", "user", "pc", "filename", "content"),
    "http": ("id", "date", "user", "pc", "url", "content"),
    "email": ("id", "date", "user", "pc", "to", "cc", "bcc", "from", "size",
              "attachments", "content"),
}

# Monthly LDAP snapshots: data/raw/r4.2/LDAP/YYYY-MM.csv
LDAP_HEADER: tuple[str, ...] = (
    "employee_name", "user_id", "email", "role", "business_unit",
    "functional_unit", "department", "team", "supervisor",
)

# CERT timestamps are US-format local time with no zone: "01/02/2010 07:11:03".
CERT_TIME_FORMAT = "%m/%d/%Y %H:%M:%S"

# The four activity files the paper replays. email.csv is parsed but not mapped:
# the paper's mapping covers logon, file, device and http only.
MAPPED_SOURCES: tuple[str, ...] = ("logon", "device", "file", "http")


class SchemaError(ValueError):
    """A CERT file's header does not match the documented r4.2 layout."""


def validate_header(source: str, header: list[str]) -> None:
    """Raise :class:`SchemaError` naming the mismatch, instead of mis-parsing silently."""
    expected = CERT_HEADERS.get(source)
    if expected is None:
        raise SchemaError(f"unknown CERT source {source!r}")
    actual = tuple(h.strip() for h in header)
    if actual != expected:
        raise SchemaError(
            f"{source}.csv header mismatch\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "If this corpus is a different CERT release, correct CERT_HEADERS in "
            "ztb/features/schema.py rather than working around it downstream."
        )


def parse_cert_time(value: str) -> datetime:
    return datetime.strptime(value.strip(), CERT_TIME_FORMAT)


# --- the normalised event we emit -----------------------------------------


@dataclass(slots=True)
class CloudEvent:
    """A CERT row rendered as a CloudTrail-style control-plane event.

    Field names follow CloudTrail loosely (eventTime, eventName, userIdentity,
    resources) so the mapping is legible to anyone who knows the AWS log format.
    """

    event_id: str            # CERT row id, preserved so answers/ labels can be joined
    ts: datetime             # original CERT timestamp, unmodified
    principal: str           # CERT user id, e.g. "AAM0658"
    action: str              # e.g. "s3:GetObject", "sts:AssumeRole"
    resource: str            # ARN-style resource string
    source: str              # originating CERT file: logon | device | file | http
    pc: str = ""             # originating host, used for device/network features
    egress: bool = False     # removable-media / external-proxy marker
    bytes_read: int = 0      # size proxy, 0 where CERT records none
    label: int = 0           # 1 if this event id appears in answers/
    extra: dict[str, str] = field(default_factory=dict)

    def sort_key(self) -> tuple[datetime, str]:
        return (self.ts, self.event_id)
