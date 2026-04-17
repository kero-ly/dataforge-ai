# tests/test_dead_letter.py
"""Tests for the Dead Letter Queue feature."""
import json
import tempfile
from pathlib import Path

from dataforge.evaluators.base import BaseEvaluator
from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy


class AlwaysFailStrategy(BaseStrategy):
    """Strategy that always raises an error."""

    async def apply(self, record: DataRecord) -> DataRecord:
        raise RuntimeError(f"Intentional failure for {record.id}")


class PassStrategy(BaseStrategy):
    """Strategy that always succeeds."""

    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": "ok"}
        return record


class AlwaysRejectEvaluator(BaseEvaluator):
    """Evaluator that always rejects."""

    async def evaluate(self, record: DataRecord) -> bool:
        return False


def write_input(path: Path, n: int) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"instruction": f"q{i}", "id": f"dl-{i}"}) + "\n")


async def test_dead_letter_captures_failed_records():
    """Failed records should be written to the dead letter file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        dl_path = Path(tmpdir) / "dead_letter.jsonl"

        write_input(input_path, 3)

        pipeline = Pipeline(
            strategy=AlwaysFailStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            max_retries=0,
            dead_letter_path=str(dl_path),
        )
        result = await pipeline.run(
            str(input_path), str(output_path), concurrency=2, show_progress=False,
        )

        assert result.failed == 3
        assert result.completed == 0

        # Dead letter file should contain all 3 failed records
        dl_lines = dl_path.read_text().strip().splitlines()
        assert len(dl_lines) == 3
        for line in dl_lines:
            data = json.loads(line)
            assert data["status"] == "FAILED"
            assert "error" in data["metadata"]


async def test_dead_letter_not_created_without_config():
    """Without dead_letter_path, no dead letter file should be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        dl_path = Path(tmpdir) / "dead_letter.jsonl"

        write_input(input_path, 2)

        pipeline = Pipeline(
            strategy=AlwaysFailStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            max_retries=0,
        )
        await pipeline.run(
            str(input_path), str(output_path), concurrency=2, show_progress=False,
        )

        assert not dl_path.exists()


async def test_dead_letter_empty_when_all_pass():
    """When all records pass, dead letter file should be empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        dl_path = Path(tmpdir) / "dead_letter.jsonl"

        write_input(input_path, 5)

        pipeline = Pipeline(
            strategy=PassStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            dead_letter_path=str(dl_path),
        )
        result = await pipeline.run(
            str(input_path), str(output_path), concurrency=2, show_progress=False,
        )

        assert result.completed == 5
        # Dead letter file either doesn't exist or is empty
        if dl_path.exists():
            assert dl_path.read_text().strip() == ""
