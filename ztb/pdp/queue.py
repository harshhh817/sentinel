"""Asynchronous ledger queue and committer -- line 11 of Algorithm 1.

The PDP computes the record digest and signature inline, enqueues, and returns; a
committer task drains the queue off the critical path (G6). The sink is pluggable:
until Module 4 it is an append-only JSONL file standing in for ``LogAccess``; the
Fabric client replaces it without touching the PDP.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Protocol


class LedgerSink(Protocol):
    def commit(self, record: dict[str, Any]) -> None: ...
    def read_all(self) -> list[dict[str, Any]]: ...


class JsonlLedger:
    """Append-only local ledger. No update or delete operations exist here either."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def commit(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock, self.path.open("a") as fh:
            fh.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open() as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def by_principal(self, principal: str) -> list[dict[str, Any]]:
        recs = [r for r in self.read_all() if r["principal"] == principal]
        return sorted(recs, key=lambda r: r["seq"])

    def by_resource(self, resource: str) -> list[dict[str, Any]]:
        return [r for r in self.read_all() if r["resource"] == resource]


class LedgerQueue:
    """asyncio.Queue plus a committer task draining into the sink."""

    def __init__(self, sink: LedgerSink, maxsize: int = 100_000):
        self.sink = sink
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._task: asyncio.Task | None = None
        self.committed = 0
        self.failed = 0

    def enqueue(self, record: dict[str, Any]) -> None:
        self._q.put_nowait(record)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            rec = await self._q.get()
            try:
                await asyncio.to_thread(self.sink.commit, rec)
                self.committed += 1
            except Exception:  # noqa: BLE001 - the committer must never die
                self.failed += 1
            finally:
                self._q.task_done()

    async def flush(self) -> None:
        await self._q.join()

    async def stop(self) -> None:
        await self.flush()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def pending(self) -> int:
        return self._q.qsize()
