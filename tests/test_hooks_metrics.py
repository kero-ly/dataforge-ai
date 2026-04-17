# tests/test_hooks_metrics.py
"""Tests for pipeline hooks, MetricsCollector, and PipelineResult."""
import json
import tempfile
from pathlib import Path

from dataforge.evaluators.base import BaseEvaluator
from dataforge.hooks import PipelineHook
from dataforge.metrics import PipelineResult
from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy

# --- Test helpers ---

class EchoStrategy(BaseStrategy):
    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": "generated"}
        return record


class RejectFilter(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        return False


class BrokenStrategy(BaseStrategy):
    async def apply(self, record: DataRecord) -> DataRecord:
        raise RuntimeError("boom")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# --- PipelineResult unit tests ---

def test_pipeline_result_summary():
    result = PipelineResult(
        total_records=10, completed=7, rejected=2, failed=1,
        elapsed_seconds=5.0, records_per_second=2.0, scores=[4.0, 5.0, 3.0],
    )
    s = result.summary()
    assert "10" in s
    assert "7" in s
    assert "2" in s
    assert "1" in s
    assert "5.0s" in s
    assert "2.00 rec/s" in s
    assert "4.00" in s  # avg score


def test_pipeline_result_avg_score_none_when_empty():
    result = PipelineResult()
    assert result.avg_score is None


def test_pipeline_result_avg_score():
    result = PipelineResult(scores=[3.0, 5.0])
    assert result.avg_score == 4.0


# --- Hook tests ---

class RecordingHook(PipelineHook):
    """Hook that records all events for assertion."""
    def __init__(self) -> None:
        self.events: list[str] = []
        self.started_ids: list[str] = []
        self.completed_ids: list[str] = []
        self.rejected_ids: list[str] = []
        self.failed_ids: list[str] = []

    async def on_pipeline_start(self, *, input_path: str, output_path: str, concurrency: int) -> None:
        self.events.append("start")

    async def on_record_started(self, record: DataRecord) -> None:
        self.events.append("started")
        self.started_ids.append(record.id)

    async def on_record_completed(self, record: DataRecord) -> None:
        self.events.append("completed")
        self.completed_ids.append(record.id)

    async def on_record_rejected(self, record: DataRecord) -> None:
        self.events.append("rejected")
        self.rejected_ids.append(record.id)

    async def on_record_failed(self, record: DataRecord, error: Exception) -> None:
        self.events.append("failed")
        self.failed_ids.append(record.id)

    async def on_pipeline_end(self) -> None:
        self.events.append("end")


async def test_hooks_called_on_completed_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [{"instruction": "q1"}, {"instruction": "q2"}])

        hook = RecordingHook()
        pipeline = Pipeline(
            strategy=EchoStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            hooks=[hook],
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=2)

        assert "start" in hook.events
        assert "end" in hook.events
        assert hook.events.count("started") == 2
        assert hook.events.count("completed") == 2
        assert len(hook.started_ids) == 2
        assert len(hook.completed_ids) == 2
        assert result.completed == 2


async def test_hooks_called_on_rejected_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [{"instruction": "reject me"}])

        hook = RecordingHook()
        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[RejectFilter()],
            checkpoint_dir=f"{tmpdir}/ckpt",
            hooks=[hook],
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=1)

        assert hook.events.count("rejected") == 1
        assert result.rejected == 1
        assert result.completed == 0


async def test_hooks_called_on_failed_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [{"instruction": "fail me"}])

        hook = RecordingHook()
        pipeline = Pipeline(
            strategy=BrokenStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            max_retries=0,
            hooks=[hook],
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=1)

        assert hook.events.count("failed") == 1
        assert result.failed == 1


async def test_pipeline_returns_pipeline_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [{"instruction": f"q{i}"} for i in range(3)])

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=2)

        assert isinstance(result, PipelineResult)
        assert result.total_records == 3
        assert result.completed == 3
        assert result.rejected == 0
        assert result.failed == 0
        assert result.elapsed_seconds > 0
        assert result.records_per_second > 0


async def test_metrics_collector_auto_added():
    """If no MetricsCollector hook is provided, one is auto-created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [{"instruction": "test"}])

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            hooks=[],  # explicitly empty
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=1)

        assert result.completed == 1


async def test_hook_exception_does_not_crash_pipeline():
    """A broken hook must not stop the pipeline."""
    class BrokenHook(PipelineHook):
        async def on_record_completed(self, record: DataRecord) -> None:
            raise RuntimeError("hook exploded")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [{"instruction": "test"}])

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            hooks=[BrokenHook()],
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=1)

        # Pipeline should complete despite broken hook
        assert result.completed == 1
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 1


async def test_completed_hook_runs_after_output_write():
    class OutputVisibleHook(PipelineHook):
        def __init__(self, output_path: Path) -> None:
            self.output_path = output_path
            self.seen = False

        async def on_record_completed(self, record: DataRecord) -> None:
            lines = self.output_path.read_text().strip().splitlines()
            ids = {json.loads(line)["id"] for line in lines}
            self.seen = record.id in ids

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        write_jsonl(input_path, [{"instruction": "visible"}])

        hook = OutputVisibleHook(output_path)
        pipeline = Pipeline(
            strategy=EchoStrategy(),
            checkpoint_dir=f"{tmpdir}/ckpt",
            hooks=[hook],
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=1)

        assert result.completed == 1
        assert hook.seen is True


async def test_mixed_results_metrics():
    """Test metrics with a mix of completed and rejected records."""
    class SelectiveFilter(BaseEvaluator):
        async def evaluate(self, record: DataRecord) -> bool:
            return "pass" in record.seed_data.get("instruction", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"

        write_jsonl(input_path, [
            {"instruction": "pass this"},
            {"instruction": "reject this"},
            {"instruction": "pass also"},
        ])

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[SelectiveFilter()],
            checkpoint_dir=f"{tmpdir}/ckpt",
        )
        result = await pipeline.run(str(input_path), str(output_path), concurrency=2)

        assert result.completed == 2
        assert result.rejected == 1
        assert result.total_records == 3
