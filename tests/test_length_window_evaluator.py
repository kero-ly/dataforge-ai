from dataforge.evaluators.length_window import LengthWindowEvaluator
from dataforge.schema import DataRecord


async def test_length_window_evaluator_enforces_windows():
    evaluator = LengthWindowEvaluator(
        min_instruction_tokens=1,
        max_instruction_tokens=4,
        min_response_tokens=2,
        max_response_tokens=5,
    )
    record = DataRecord(
        seed_data={"instruction": "tiny prompt"},
        synthetic_data={"response": "short"},
    )

    result = await evaluator.assess(record)

    assert result.passed is False
    assert "response_too_short" in result.reason_codes
