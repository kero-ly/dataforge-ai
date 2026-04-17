from dataforge.metrics import MetricsCollector
from dataforge.schema import DataRecord


async def test_metrics_collector_tracks_assessment_breakdown():
    collector = MetricsCollector()
    await collector.on_pipeline_start(input_path="in", output_path="out", concurrency=1)
    record = DataRecord(
        seed_data={"instruction": "q"},
        synthetic_data={"response": "a"},
        metadata={
            "assessment": {
                "results": [{"evaluator": "DemoEval", "passed": True, "score": 1.0}],
                "aggregate_score": 1.0,
                "passed_all": True,
            }
        },
    )
    await collector.on_record_completed(record)
    await collector.on_pipeline_end()

    assert collector.result is not None
    assert collector.result.assessment_avg_score == 1.0
    assert collector.result.evaluator_breakdown["DemoEval"]["passed"] == 1
