# src/dataforge/engine/checkpoint.py
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import aiofiles

if TYPE_CHECKING:
    from aiofiles.threadpool.text import AsyncTextIOWrapper

logger = logging.getLogger(__name__)


class CheckpointManager:
    """WAL-based checkpoint for zero data loss on restart.

    Each completed record ID is appended to a JSONL file (the WAL).
    On restart, load() reads the file and rebuilds the in-memory set of done IDs.
    commit() appends immediately (no buffering) to guarantee durability.

    Supports async context manager for persistent file handle during pipeline runs::

        mgr = CheckpointManager(checkpoint_dir="./runs/experiment_1")
        await mgr.load()
        async with mgr:
            # commit() uses persistent file handle — much faster
            await mgr.commit(record.id)

    Also works without context manager (opens/closes file per commit, slower).
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._dir = Path(checkpoint_dir)
        self._wal_path = self._dir / "checkpoint.jsonl"
        self._done: set[str] = set()
        self._lock = asyncio.Lock()
        self._wal_handle: AsyncTextIOWrapper | None = None

    async def __aenter__(self) -> CheckpointManager:
        """Open persistent WAL file handle for the duration of the pipeline."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._wal_handle = await aiofiles.open(self._wal_path, "a", encoding="utf-8")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the persistent WAL file handle."""
        if self._wal_handle is not None:
            await self._wal_handle.close()
            self._wal_handle = None

    async def load(self) -> None:
        """Load completed IDs from disk. Call once at pipeline startup.

        Creates the checkpoint directory if it doesn't exist yet.
        Safe to call even if no checkpoint file exists.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        if not self._wal_path.exists():
            logger.debug("No checkpoint file found at %s, starting fresh", self._wal_path)
            return
        async with aiofiles.open(self._wal_path, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._done.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed checkpoint line: %s", line.strip()[:200])

    async def commit(self, record_id: str) -> None:
        """Append a completed record ID to the WAL file (write-ahead log).

        Uses the persistent file handle if available (async context manager),
        otherwise opens/closes the file per call.
        """
        async with self._lock:
            if record_id in self._done:
                return  # already committed; skip the disk write
            self._done.add(record_id)
            payload = json.dumps({"id": record_id}) + "\n"
            if self._wal_handle is not None:
                await self._wal_handle.write(payload)
                await self._wal_handle.flush()
            else:
                self._dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(self._wal_path, "a", encoding="utf-8") as f:
                    await f.write(payload)

    async def commit_batch(self, record_ids: list[str]) -> None:
        """Append multiple completed record IDs in a single WAL write + flush.

        More efficient than calling commit() in a loop because it amortises
        the disk flush across all IDs in the batch.
        """
        async with self._lock:
            new_ids = [rid for rid in record_ids if rid not in self._done]
            if not new_ids:
                return
            self._done.update(new_ids)
            payload = "".join(json.dumps({"id": rid}) + "\n" for rid in new_ids)
            if self._wal_handle is not None:
                await self._wal_handle.write(payload)
                await self._wal_handle.flush()
            else:
                self._dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(self._wal_path, "a", encoding="utf-8") as f:
                    await f.write(payload)

    async def is_done(self, record_id: str) -> bool:
        """Return True if this record ID has been committed.

        Lock-free: safe in single-threaded asyncio because ``_done`` is only
        mutated inside ``commit``/``commit_batch`` (which hold the lock), and
        ``set.__contains__`` is an atomic read in CPython.
        """
        return record_id in self._done

    async def load_done_ids(self) -> set[str]:
        """Return a snapshot of all completed IDs currently in memory."""
        return set(self._done)

    @property
    def completed_count(self) -> int:
        """Number of completed record IDs currently tracked in memory."""
        return len(self._done)
