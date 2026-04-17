from dataforge.assessment.schema import AssessmentResult, DatasetAssessmentSummary, RecordAssessment
from dataforge.schema import DataRecord


def test_assessment_schema_models_round_trip():
    record = DataRecord(seed_data={"instruction": "Hi"}, synthetic_data={"response": "Hello"})
    result = AssessmentResult(evaluator="demo", passed=True, score=1.0)
    assessment = RecordAssessment(
        record_id="rec-1",
        line_number=1,
        source_path="sample.jsonl",
        normalized_record=record,
        results=[result],
        aggregate_score=1.0,
        passed_all_required=True,
    )
    summary = DatasetAssessmentSummary(
        suite="sft_readiness_v1",
        total_records=1,
        sampled_records=1,
        sample_seed=42,
        overall_quality_score=100.0,
        evaluator_summaries=[],
        dataset_metrics={},
    )

    assert assessment.model_dump()["record_id"] == "rec-1"
    assert summary.model_dump()["overall_quality_score"] == 100.0
