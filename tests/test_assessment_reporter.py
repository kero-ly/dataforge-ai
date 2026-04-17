import json
import tempfile
from pathlib import Path

from dataforge.assessment.reporter import write_assessment_report
from dataforge.assessment.schema import DatasetAssessmentSummary


def test_assessment_reporter_writes_expected_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = DatasetAssessmentSummary(
            suite="sft_readiness_v1",
            total_records=10,
            sampled_records=5,
            sample_seed=42,
            overall_quality_score=88.5,
            evaluator_summaries=[],
            dataset_metrics={"duplicate_rate": 0.1},
        )
        output_dir = write_assessment_report(
            run_name="demo",
            suite_name="sft_readiness_v1",
            source_path="demo.jsonl",
            summary=summary,
            records=[],
            output_dir=tmpdir,
            output_formats=["json", "md", "html"],
            persist_record_results=True,
            config_snapshot={"name": "demo"},
        )

        assert json.loads((output_dir / "summary.json").read_text())["suite"] == "sft_readiness_v1"
        assert (output_dir / "report.md").exists()
        assert (output_dir / "report.html").exists()
