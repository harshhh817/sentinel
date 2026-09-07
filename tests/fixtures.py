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

from ztb.features.schema import CERT_HEADERS, CERT_TIME_FORMAT, LDAP_HEADER

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
                # Alternate removable / local so the read-write ratio is not degenerate.
                fname = "R:\\payroll.xlsx" if slot % 8 == 2 else "C:\\work\\notes.doc"
                rows["file"].append([eid, stamp, user, pc, fname, "file-content-" * 3])
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
        writer = csv.writer(fh)
        writer.writerow(["id", "date", "user", "pc", "activity"])
        for eid in malicious_ids:
            writer.writerow([eid, "01/01/2010 00:00:00", "CDE1846", "PC-0001", "scenario"])
    with (root / "answers" / "insiders.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "scenario", "user", "start", "end"])
        writer.writerow(["r4.2", "2", "CDE1846", "01/01/2010", "12/31/2010"])

    return {
        "root": root,
        "total_events": len(all_ids),
        "malicious_ids": set(malicious_ids),
        "users": set(USERS),
        "rows_per_source": {k: len(v) for k, v in rows.items()},
        "start": START,
        "days": days,
    }
