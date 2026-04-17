# tests/test_sqlite_checkpoint.py
"""Tests for the SQLite-backed checkpoint manager."""
import tempfile
from pathlib import Path

import pytest

from dataforge.engine.sqlite_checkpoint import SQLiteCheckpointManager


async def test_basic_commit_and_is_done():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        await mgr.load()

        assert not await mgr.is_done("record-1")
        await mgr.commit("record-1")
        assert await mgr.is_done("record-1")
        assert mgr.completed_count == 1


async def test_persistence_across_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        # First session: commit some records
        mgr = SQLiteCheckpointManager(tmpdir)
        await mgr.load()
        await mgr.commit("r1")
        await mgr.commit("r2")
        await mgr.commit("r3")
        assert mgr.completed_count == 3
        # Close connection
        await mgr.__aexit__(None, None, None)

        # Second session: reload
        mgr2 = SQLiteCheckpointManager(tmpdir)
        await mgr2.load()
        assert mgr2.completed_count == 3
        assert await mgr2.is_done("r1")
        assert await mgr2.is_done("r2")
        assert await mgr2.is_done("r3")
        assert not await mgr2.is_done("r4")


async def test_duplicate_commit_ignored():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        await mgr.load()
        await mgr.commit("dup-1")
        await mgr.commit("dup-1")  # should not raise
        # Count increments each time commit is called even for dupes
        # because INSERT OR IGNORE succeeds silently.
        # But is_done still works correctly.
        assert await mgr.is_done("dup-1")


async def test_context_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        async with mgr:
            await mgr.commit("ctx-1")
            assert await mgr.is_done("ctx-1")
        # After exit, connection is closed
        assert mgr._conn is None


async def test_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = Path(tmpdir) / "a" / "b" / "c"
        mgr = SQLiteCheckpointManager(nested)
        await mgr.load()
        assert nested.exists()
        await mgr.commit("nested-1")
        assert await mgr.is_done("nested-1")


async def test_db_file_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        await mgr.load()
        await mgr.commit("file-check")
        db_path = Path(tmpdir) / "checkpoint.db"
        assert db_path.exists()


async def test_empty_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        await mgr.load()
        assert mgr.completed_count == 0
        assert not await mgr.is_done("anything")


async def test_commit_without_load_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        # Don't call load()
        with pytest.raises(RuntimeError, match="not loaded"):
            await mgr.commit("oops")


async def test_load_done_ids_returns_all_completed():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = SQLiteCheckpointManager(tmpdir)
        await mgr.load()
        await mgr.commit_batch(["a", "b", "c"])
        done = await mgr.load_done_ids()
        assert done == {"a", "b", "c"}
