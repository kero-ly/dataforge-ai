# tests/test_performance.py
"""Performance and stress tests for the pipeline."""
import json
import tempfile
import time
from pathlib import Path

from dataforge.evaluators.base import BaseEvaluator
from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy


class FastStrategy(BaseStrategy):
    """No-op strategy for throughput testing."""

    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": "ok"}
        return record


class FastEvaluator(BaseEvaluator):
    """Always-pass evaluator for throughput testing."""

    async def evaluate(self, record: DataRecord) -> bool:
        return True


def write_jsonl(path: Path, n: int) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"instruction": f"q{i}", "id": f"id-{i:06d}"}) + "\n")


async def test_throughput_1000_records():
    """Process 1000 records and ensure reasonable throughput."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, 1000)

        pipeline = Pipeline(
            strategy=FastStrategy(),
            evaluators=[FastEvaluator()],
            checkpoint_dir=f"{tmpdir}/ckpt",
        )

        start = time.monotonic()
        result = await pipeline.run(
            str(input_path), str(output_path), concurrency=20, show_progress=False,
        )
        elapsed = time.monotonic() - start

        assert result.completed == 1000
        assert result.total_records == 1000
        assert result.failed == 0
        # Should complete in under 10 seconds for 1000 no-op records
        assert elapsed < 10.0, f"Pipeline too slow: {elapsed:.1f}s for 1000 records"


async def test_checkpoint_resume_correctness_at_scale():
    """Checkpoint must not produce duplicates after resume with 500 records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, 500)

        pipeline1 = Pipeline(
            strategy=FastStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
        )
        await pipeline1.run(str(input_path), str(output_path), concurrency=10, show_progress=False)

        first_count = len(output_path.read_text().strip().splitlines())
        assert first_count == 500

        # Second run: everything is checkpointed, no new output
        pipeline2 = Pipeline(
            strategy=FastStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
        )
        await pipeline2.run(str(input_path), str(output_path), concurrency=10, show_progress=False)

        all_lines = output_path.read_text().strip().splitlines()
        all_ids = [json.loads(line)["id"] for line in all_lines]
        assert len(all_ids) == len(set(all_ids)), "Duplicate records after resume!"


async def test_high_concurrency():
    """Pipeline should handle concurrency=100 without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, 200)

        pipeline = Pipeline(
            strategy=FastStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
        )
        result = await pipeline.run(
            str(input_path), str(output_path), concurrency=100, show_progress=False,
        )

        assert result.completed == 200
        assert result.failed == 0


async def test_empty_input_file():
    """Pipeline should handle an empty input file gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        input_path.write_text("")

        pipeline = Pipeline(
            strategy=FastStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
        )
        result = await pipeline.run(
            str(input_path), str(output_path), concurrency=5, show_progress=False,
        )

        assert result.completed == 0
        assert result.total_records == 0
