# src/dataforge/engine/sqlite_checkpoint.py
"""SQLite-backed checkpoint manager for million-scale datasets.

Drop-in replacement for :class:`CheckpointManager` when keeping all
completed IDs in an in-memory set becomes impractical. Uses SQLite's
WAL journal mode for high write throughput and supports concurrent reads.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)


class SQLiteCheckpointManager:
    """SQLite-backed checkpoint for zero data loss on restart.

    Stores completed record IDs in a SQLite database instead of an
    in-memory set. This scales to millions of records with constant
    memory usage.

    Supports the same async context manager interface as CheckpointManager::

        mgr = SQLiteCheckpointManager(checkpoint_dir="./runs/exp_1")
        await mgr.load()
        async with mgr:
            await mgr.commit(record.id)
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._dir = Path(checkpoint_dir)
        self._db_path = self._dir / "checkpoint.db"
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._count = 0

    async def __aenter__(self) -> SQLiteCheckpointManager:
        """Ensure database is open for the duration of the pipeline."""
        if self._conn is None:
            await self.load()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def load(self) -> None:
        """Open the database and create tables if needed.

        Safe to call even if no database file exists yet.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()
        self._conn = await loop.run_in_executor(None, self._open_db)
        self._count = await loop.run_in_executor(None, self._count_rows)
        if self._count > 0:
            logger.info(
                "SQLite checkpoint loaded: %d records already completed",
                self._count,
            )

    def _open_db(self) -> sqlite3.Connection:
        """Open SQLite connection with WAL mode for concurrent access."""
        conn = sqlite3.connect(
            str(self._db_path), timeout=30, check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS completed (id TEXT PRIMARY KEY)"
        )
        conn.commit()
        return conn

    def _count_rows(self) -> int:
        """Count completed rows in the database."""
        if self._conn is None:
            return 0
        cursor = self._conn.execute("SELECT COUNT(*) FROM completed")
        return cursor.fetchone()[0]

    async def commit(self, record_id: str) -> None:
        """Record a completed ID in the database."""
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Checkpoint not loaded; call load() first")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self._insert, record_id
            )
            self._count += 1

    async def commit_batch(self, record_ids: list[str]) -> None:
        """Record multiple completed IDs in a single transaction.

        More efficient than calling commit() in a loop because it uses a
        single transaction and single commit for all IDs.
        """
        async with self._lock:
            if self._conn is None:
                raise RuntimeError("Checkpoint not loaded; call load() first")
            loop = asyncio.get_running_loop()
            inserted = await loop.run_in_executor(
                None, self._insert_batch, record_ids
            )
            self._count += inserted

    def _insert(self, record_id: str) -> None:
        """Insert a record ID into the database (synchronous)."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO completed (id) VALUES (?)",
            (record_id,),
        )
        self._conn.commit()

    def _insert_batch(self, record_ids: list[str]) -> int:
        """Insert multiple record IDs in a single transaction (synchronous)."""
        if self._conn is None:
            return 0
        self._conn.executemany(
            "INSERT OR IGNORE INTO completed (id) VALUES (?)",
            [(rid,) for rid in record_ids],
        )
        self._conn.commit()
        return len(record_ids)

    async def is_done(self, record_id: str) -> bool:
        """Check whether a record ID has been committed."""
        async with self._lock:
            if self._conn is None:
                return False
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._exists, record_id
            )

    async def load_done_ids(self) -> set[str]:
        """Load all completed IDs as a set for fast in-memory membership checks."""
        async with self._lock:
            if self._conn is None:
                return set()
            loop = asyncio.get_running_loop()
            rows = await loop.run_in_executor(None, self._fetch_all_ids)
            return set(rows)

    def _exists(self, record_id: str) -> bool:
        """Check existence in database (synchronous)."""
        if self._conn is None:
            return False
        cursor = self._conn.execute(
            "SELECT 1 FROM completed WHERE id = ?", (record_id,)
        )
        return cursor.fetchone() is not None

    def _fetch_all_ids(self) -> list[str]:
        """Fetch all completed IDs from SQLite (synchronous)."""
        if self._conn is None:
            return []
        cursor = self._conn.execute("SELECT id FROM completed")
        return [row[0] for row in cursor.fetchall()]

    @property
    def completed_count(self) -> int:
        """Number of completed record IDs."""
        return self._count
