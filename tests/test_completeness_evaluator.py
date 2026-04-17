from dataforge.evaluators.completeness import CompletenessEvaluator
from dataforge.schema import DataRecord


async def test_completeness_evaluator_reports_missing_fields():
    evaluator = CompletenessEvaluator()
    record = DataRecord(seed_data={"instruction": ""}, synthetic_data={"response": ""})

    result = await evaluator.assess(record)

    assert result.passed is False
    assert "missing_instruction" in result.reason_codes
    assert "missing_response" in result.reason_codes
