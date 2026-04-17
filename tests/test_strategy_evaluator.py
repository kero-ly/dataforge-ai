# tests/test_strategy_evaluator.py
import pytest

from dataforge.evaluators.base import BaseEvaluator
from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy

# --- Concrete test implementations ---

class EchoStrategy(BaseStrategy):
    """Test strategy: copies seed_data to synthetic_data unchanged."""

    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = dict(record.seed_data)
        return record


class AlwaysPassEvaluator(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        return True


class AlwaysRejectEvaluator(BaseEvaluator):
    async def evaluate(self, record: DataRecord) -> bool:
        return False


# --- Tests ---

async def test_strategy_apply():
    strategy = EchoStrategy()
    record = DataRecord(seed_data={"question": "What is 2+2?"})
    result = await strategy.apply(record)
    assert result.synthetic_data == {"question": "What is 2+2?"}


async def test_evaluator_pass():
    evaluator = AlwaysPassEvaluator()
    record = DataRecord(seed_data={"q": "test"})
    assert await evaluator.evaluate(record) is True


async def test_evaluator_reject():
    evaluator = AlwaysRejectEvaluator()
    record = DataRecord(seed_data={"q": "test"})
    assert await evaluator.evaluate(record) is False


def test_base_strategy_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseStrategy()  # type: ignore


def test_base_evaluator_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseEvaluator()  # type: ignore


async def test_strategy_returns_same_record():
    """Strategy must return the DataRecord, not a new object or None."""
    strategy = EchoStrategy()
    record = DataRecord(seed_data={"x": 1})
    result = await strategy.apply(record)
    assert result is record  # same object identity


async def test_evaluator_returns_bool():
    """Evaluator must return a bool, not truthy/falsy value."""
    pass_eval = AlwaysPassEvaluator()
    reject_eval = AlwaysRejectEvaluator()
    record = DataRecord(seed_data={"x": 1})
    pass_result = await pass_eval.evaluate(record)
    reject_result = await reject_eval.evaluate(record)
    assert type(pass_result) is bool
    assert type(reject_result) is bool
