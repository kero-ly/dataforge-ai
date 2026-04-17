from dataforge.evaluators.base import BaseEvaluator
from dataforge.schema import DataRecord


class CompatEvaluator(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        record.score = 4.0
        return True


async def test_base_evaluator_assess_wraps_legacy_evaluate():
    evaluator = CompatEvaluator()
    record = DataRecord(seed_data={"instruction": "Hello"}, synthetic_data={"response": "World"})

    result = await evaluator.assess(record)

    assert result.passed is True
    assert result.score == 4.0
    assert result.evaluator == "CompatEvaluator"
