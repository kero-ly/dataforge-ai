import json
import tempfile
from pathlib import Path

from dataforge.config.assessment_schema import (
    AssessmentConfig,
    AssessmentOutputConfig,
    AssessmentSourceConfig,
    AssessmentSuiteConfig,
)
from dataforge.assessment.runner import AssessmentRunner


async def test_assessment_runner_generates_summary_and_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.jsonl"
        data_path.write_text(
            "\n".join(
                [
                    json.dumps({"instruction": "Say hi", "response": "Hello there"}),
                    json.dumps({"instruction": "List two fruits", "response": "- apple\n- pear"}),
                ]
            ),
            encoding="utf-8",
        )
        config = AssessmentConfig(
            name="demo-assessment",
            source=AssessmentSourceConfig(path=str(data_path)),
            suite=AssessmentSuiteConfig(name="sft_readiness_v1", sample_size=2, sample_seed=7),
            output=AssessmentOutputConfig(dir=tmpdir, formats=["json", "md", "html"]),
        )

        output_dir, summary = await AssessmentRunner(config).run()

        assert summary.sampled_records == 2
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "report.md").exists()
        assert (output_dir / "report.html").exists()
