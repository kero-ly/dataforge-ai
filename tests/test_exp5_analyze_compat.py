import json
import tempfile
from pathlib import Path

from experiments.exp5_quality.analyze_quality import load_judge_results, load_mt_bench_results


def test_exp5_analyzer_loads_new_summary_layout():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        assessment_summary = root / "raw_seed" / "assessment" / "raw_seed-assessment" / "20260310_120000" / "summary.json"
        assessment_summary.parent.mkdir(parents=True, exist_ok=True)
        assessment_summary.write_text(
            json.dumps(
                {
                    "overall_quality_score": 80.0,
                    "sampled_records": 20,
                    "evaluator_summaries": [
                        {"evaluator": "MultiCriteriaEvaluator", "pass_rate": 0.6}
                    ],
                }
            ),
            encoding="utf-8",
        )

        benchmark_summary = root / "raw_seed" / "benchmark" / "raw_seed-benchmark" / "20260310_120000" / "summary.json"
        benchmark_summary.parent.mkdir(parents=True, exist_ok=True)
        benchmark_summary.write_text(
            json.dumps(
                {
                    "task_summaries": [
                        {
                            "task": "mt_bench_lite_v1",
                            "category_scores": {"math": 80.0},
                            "overall_score": 80.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        judge = load_judge_results(root)
        mt = load_mt_bench_results(root)

        assert judge["raw_seed"]["avg_score"] == 4.0
        assert mt["raw_seed"]["overall_score"] == 8.0
