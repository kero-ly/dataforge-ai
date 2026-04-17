# tests/test_pipeline.py
import json
import tempfile
from pathlib import Path

from dataforge.clients.base import BaseLLMClient
from dataforge.evaluators.base import BaseEvaluator
from dataforge.pipeline import Pipeline, _plan_zero_overhead_batch
from dataforge.schema import DataRecord, RecordStatus
from dataforge.strategies.base import BaseStrategy

# --- Test helpers ---

class MockClient(BaseLLMClient):
    async def generate(self, prompt, **kwargs):
        return "mock"


class EchoStrategy(BaseStrategy):
    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": "generated"}
        return record


class PassFilter(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        return True


class RejectFilter(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        return False


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# --- Tests ---

async def test_pipeline_processes_all_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        seeds = [{"instruction": f"question {i}"} for i in range(5)]
        write_jsonl(input_path, seeds)

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[PassFilter()],
            checkpoint_dir=str(checkpoint_dir),
        )
        await pipeline.run(
            input_path=str(input_path),
            output_path=str(output_path),
            concurrency=3,
        )

        output_lines = output_path.read_text().strip().splitlines()
        assert len(output_lines) == 5
        for line in output_lines:
            record = DataRecord.model_validate_json(line)
            assert record.status == RecordStatus.COMPLETED
            assert record.synthetic_data == {"response": "generated"}


async def test_pipeline_rejects_filtered_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        seeds = [{"instruction": "test"}]
        write_jsonl(input_path, seeds)

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[RejectFilter()],
            checkpoint_dir=str(checkpoint_dir),
        )
        await pipeline.run(
            input_path=str(input_path),
            output_path=str(output_path),
            concurrency=2,
        )

        # Rejected records should NOT appear in output.jsonl
        assert not output_path.exists() or output_path.read_text().strip() == ""


async def test_pipeline_resumes_from_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        # Seeds with explicit IDs so checkpoint can match them on resume
        seeds = [{"instruction": f"q{i}", "id": f"id-{i:03d}"} for i in range(4)]
        write_jsonl(input_path, seeds)

        pipeline1 = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(checkpoint_dir))
        await pipeline1.run(str(input_path), str(output_path), concurrency=2)

        first_run_ids = [
            json.loads(line)["id"]
            for line in output_path.read_text().strip().splitlines()
        ]
        assert len(first_run_ids) == 4

        # Second run: all records are already in checkpoint, so nothing new is processed
        pipeline2 = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(checkpoint_dir))
        await pipeline2.run(str(input_path), str(output_path), concurrency=2)

        second_run_lines = output_path.read_text().strip().splitlines()
        all_ids = [json.loads(line)["id"] for line in second_run_lines]
        assert len(all_ids) == len(set(all_ids)), "Duplicate records found after resume!"


async def test_pipeline_records_have_timestamp_in_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        write_jsonl(input_path, [{"instruction": "test"}])

        pipeline = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(checkpoint_dir))
        await pipeline.run(str(input_path), str(output_path), concurrency=1)

        line = output_path.read_text().strip()
        record = DataRecord.model_validate_json(line)
        assert "timestamp" in record.metadata


async def test_pipeline_no_evaluators_passes_all():
    """Pipeline with no evaluators should pass all records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        write_jsonl(input_path, [{"q": "a"}, {"q": "b"}, {"q": "c"}])

        pipeline = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(checkpoint_dir))
        await pipeline.run(str(input_path), str(output_path), concurrency=2)

        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 3


async def test_pipeline_worker_survives_strategy_failure():
    """Worker must not crash when strategy raises; output file stays empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        write_jsonl(input_path, [{"instruction": "test"}])

        class BrokenStrategy(BaseStrategy):
            async def apply(self, record: DataRecord) -> DataRecord:
                raise RuntimeError("intentional failure")

        pipeline = Pipeline(
            strategy=BrokenStrategy(),
            checkpoint_dir=str(checkpoint_dir),
        )
        await pipeline.run(str(input_path), str(output_path), concurrency=1)

        assert not output_path.exists() or output_path.read_text().strip() == ""


async def test_pipeline_burst_mode():
    """Burst mode processes all records via asyncio.gather + Semaphore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        seeds = [{"instruction": f"question {i}"} for i in range(10)]
        write_jsonl(input_path, seeds)

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[PassFilter()],
            checkpoint_dir=str(checkpoint_dir),
        )
        result = await pipeline.run(
            input_path=str(input_path),
            output_path=str(output_path),
            concurrency=5,
            mode="burst",
        )

        output_lines = output_path.read_text().strip().splitlines()
        assert len(output_lines) == 10
        for line in output_lines:
            record = DataRecord.model_validate_json(line)
            assert record.status == RecordStatus.COMPLETED
            assert record.synthetic_data == {"response": "generated"}
        assert result.completed == 10


async def test_pipeline_burst_mode_with_rejections():
    """Burst mode correctly handles rejected records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        seeds = [{"instruction": "test"}]
        write_jsonl(input_path, seeds)

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[RejectFilter()],
            checkpoint_dir=str(checkpoint_dir),
        )
        result = await pipeline.run(
            str(input_path), str(output_path), concurrency=1, mode="burst",
        )

        assert not output_path.exists() or output_path.read_text().strip() == ""
        assert result.completed == 0


async def test_pipeline_burst_prefers_checkpoint_done_id_snapshot(monkeypatch):
    """Burst mode should use load_done_ids snapshot instead of per-record is_done checks."""
    calls = {"is_done": 0}

    class FakeCheckpoint:
        def __init__(self, _checkpoint_dir):
            self.completed_count = 1

        async def load(self):
            return None

        async def load_done_ids(self):
            return {"id-001"}

        async def is_done(self, _record_id):
            calls["is_done"] += 1
            return False

        async def commit_batch(self, _record_ids):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr("dataforge.pipeline.CheckpointManager", FakeCheckpoint)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        write_jsonl(
            input_path,
            [
                {"instruction": "a", "id": "id-001"},
                {"instruction": "b", "id": "id-002"},
            ],
        )

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            checkpoint_dir=str(checkpoint_dir),
        )
        result = await pipeline.run(
            input_path=str(input_path),
            output_path=str(output_path),
            concurrency=2,
            mode="burst",
            show_progress=False,
        )

        assert result.completed == 1
        assert calls["is_done"] == 0


async def test_pipeline_burst_windowed_gather(monkeypatch):
    """Burst mode should gather in windows when burst_window_size is configured."""
    gather_calls = {"count": 0}
    real_gather = __import__("asyncio").gather

    async def counted_gather(*args, **kwargs):
        gather_calls["count"] += 1
        return await real_gather(*args, **kwargs)

    monkeypatch.setattr("dataforge.pipeline.asyncio.gather", counted_gather)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        seeds = [{"instruction": f"question {i}", "id": f"id-{i}"} for i in range(5)]
        write_jsonl(input_path, seeds)

        pipeline = Pipeline(
            strategy=EchoStrategy(),
            evaluators=[PassFilter()],
            checkpoint_dir=str(checkpoint_dir),
            burst_window_size=2,
        )
        result = await pipeline.run(
            input_path=str(input_path),
            output_path=str(output_path),
            concurrency=5,
            mode="burst",
            show_progress=False,
        )

        assert result.completed == 5
        assert gather_calls["count"] == 3


async def test_pipeline_burst_uses_strategy_seed_fast_path():
    """Burst mode should bypass strategy.apply() when apply_seed_data is available."""

    class SeedFastStrategy(BaseStrategy):
        async def apply(self, record: DataRecord) -> DataRecord:
            raise AssertionError("burst fast path should bypass apply()")

        async def apply_seed_data(self, seed_data: dict[str, object]) -> dict[str, object]:
            return {"response": f"fast:{seed_data['instruction']}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        seeds = [{"instruction": "question", "id": "id-fast"}]
        write_jsonl(input_path, seeds)

        pipeline = Pipeline(
            strategy=SeedFastStrategy(),
            checkpoint_dir=str(checkpoint_dir),
        )
        result = await pipeline.run(
            input_path=str(input_path),
            output_path=str(output_path),
            concurrency=1,
            mode="burst",
            show_progress=False,
        )

        assert result.completed == 1
        record = DataRecord.model_validate_json(output_path.read_text().strip())
        assert record.id == "id-fast"
        assert record.status == RecordStatus.COMPLETED
        assert record.synthetic_data == {"response": "fast:question"}
        assert "timestamp" in record.metadata


def test_plan_zero_overhead_batch_groups_prefixes_before_dispatch():
    class FakeSubClient:
        def __init__(self, name: str) -> None:
            self.name = name
            self._aclient = f"aclient-{name}"

    class FakeClusterClient:
        routing_strategy = "prefix_affinity"

        def __init__(self) -> None:
            self._clients = [FakeSubClient("a"), FakeSubClient("b")]

        def _prompt_prefix_key(self, messages: list[dict[str, str]]) -> str:
            return messages[0]["content"].split("|", 1)[0]

        def _pick_client_for_prompt(self, messages: list[dict[str, str]]) -> FakeSubClient:
            prefix = self._prompt_prefix_key(messages)
            return self._clients[0] if prefix.endswith("0") else self._clients[1]

    fast_records = [
        ("id-1", {"instruction": "one"}),
        ("id-2", {"instruction": "two"}),
        ("id-3", {"instruction": "three"}),
    ]
    prompt_data = [
        ("deepen", [{"role": "user", "content": "p1|msg-a"}]),
        ("constraints", [{"role": "user", "content": "p0|msg-b"}]),
        ("constraints", [{"role": "user", "content": "p1|msg-c"}]),
    ]

    planned, raw_callers = _plan_zero_overhead_batch(
        fast_records=fast_records,
        prompt_data=prompt_data,
        client=FakeClusterClient(),
        prefix_aware_scheduling=True,
        prefix_affinity_striping=False,
    )

    assert [record_id for record_id, _, _ in planned] == ["id-2", "id-3", "id-1"]
    assert raw_callers == ["aclient-a", "aclient-b", "aclient-b"]
