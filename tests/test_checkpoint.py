# tests/test_checkpoint.py
import json
import tempfile
from pathlib import Path

from dataforge.engine.checkpoint import CheckpointManager


async def test_commit_and_is_done():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir)
        await mgr.commit("record-001")
        assert await mgr.is_done("record-001") is True
        assert await mgr.is_done("record-999") is False


async def test_load_existing_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr1 = CheckpointManager(checkpoint_dir=tmpdir)
        await mgr1.commit("id-aaa")
        await mgr1.commit("id-bbb")

        # Simulate restart: new manager loads from disk
        mgr2 = CheckpointManager(checkpoint_dir=tmpdir)
        await mgr2.load()
        assert await mgr2.is_done("id-aaa") is True
        assert await mgr2.is_done("id-bbb") is True
        assert await mgr2.is_done("id-ccc") is False


async def test_checkpoint_persists_to_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir)
        await mgr.commit("disk-id-1")
        await mgr.commit("disk-id-2")

        # Read the WAL file directly
        wal_path = Path(tmpdir) / "checkpoint.jsonl"
        assert wal_path.exists()
        lines = wal_path.read_text().strip().splitlines()
        assert len(lines) == 2
        ids_on_disk = {json.loads(line)["id"] for line in lines}
        assert ids_on_disk == {"disk-id-1", "disk-id-2"}


async def test_no_duplicate_commits():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir)
        await mgr.commit("dup-id")
        await mgr.commit("dup-id")  # duplicate - should be a no-op

        # Only one line should exist in the WAL file
        wal_path = Path(tmpdir) / "checkpoint.jsonl"
        lines = wal_path.read_text().strip().splitlines()
        assert len(lines) == 1

        await mgr.load()
        assert await mgr.is_done("dup-id") is True


async def test_empty_checkpoint_dir_is_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = Path(tmpdir) / "a" / "b" / "c"
        mgr = CheckpointManager(checkpoint_dir=str(nested_dir))
        await mgr.load()  # Should not raise even if dir doesn't exist yet
        assert await mgr.is_done("anything") is False


async def test_load_done_ids_returns_snapshot():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir)
        await mgr.commit_batch(["id-1", "id-2"])
        done = await mgr.load_done_ids()
        assert done == {"id-1", "id-2"}
