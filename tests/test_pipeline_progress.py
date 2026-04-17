# tests/test_pipeline_progress.py
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy


class EchoStrategy(BaseStrategy):
    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": "generated"}
        return record


async def test_pipeline_run_accepts_show_progress_param():
    """Adding show_progress=False must not raise TypeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        input_path.write_text(json.dumps({"instruction": "test"}) + "\n")
        pipeline = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(Path(tmpdir) / "ck"))
        await pipeline.run(str(input_path), str(output_path), concurrency=1, show_progress=False)
        assert output_path.read_text().strip() != ""


async def test_pipeline_run_default_show_progress_does_not_raise():
    """Default call (no show_progress kwarg) must still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        input_path.write_text(json.dumps({"instruction": "test"}) + "\n")
        pipeline = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(Path(tmpdir) / "ck"))
        await pipeline.run(str(input_path), str(output_path), concurrency=1)
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 1


async def test_pipeline_progress_advance_called_per_record():
    """progress.advance() should be called once per processed record."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        n_records = 4
        with open(input_path, "w") as f:
            for i in range(n_records):
                f.write(json.dumps({"instruction": f"q{i}"}) + "\n")

        mock_progress = MagicMock()
        mock_progress.add_task = MagicMock(return_value=0)

        pipeline = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(Path(tmpdir) / "ck"))
        with patch.object(pipeline, "_make_progress", return_value=(mock_progress, 0)):
            await pipeline.run(str(input_path), str(output_path), concurrency=2, show_progress=True)

        assert mock_progress.advance.call_count == n_records


async def test_pipeline_show_progress_false_no_progress_created():
    """When show_progress=False, _make_progress must never be called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        input_path.write_text(json.dumps({"instruction": "x"}) + "\n")
        pipeline = Pipeline(strategy=EchoStrategy(), checkpoint_dir=str(Path(tmpdir) / "ck"))
        with patch.object(pipeline, "_make_progress") as mock_make:
            await pipeline.run(str(input_path), str(output_path), concurrency=1, show_progress=False)
        mock_make.assert_not_called()
