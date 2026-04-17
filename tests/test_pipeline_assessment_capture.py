import json
import tempfile
from pathlib import Path

from dataforge.evaluators.completeness import CompletenessEvaluator
from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy


class SimpleStrategy(BaseStrategy):
    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": "hello world"}
        return record


async def test_pipeline_can_capture_assessment_details():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        input_path.write_text(json.dumps({"instruction": "say hi"}) + "\n", encoding="utf-8")

        pipeline = Pipeline(
            strategy=SimpleStrategy(),
            evaluators=[CompletenessEvaluator()],
            checkpoint_dir=f"{tmpdir}/ckpt",
            capture_assessment_details=True,
        )
        await pipeline.run(str(input_path), str(output_path), concurrency=1, show_progress=False)
        row = json.loads(output_path.read_text(encoding="utf-8").strip())
        assert row["metadata"]["assessment"]["passed_all"] is True
        assert row["metadata"]["assessment"]["results"][0]["evaluator"] == "CompletenessEvaluator"
