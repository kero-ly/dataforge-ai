# tests/test_llm_judge.py
import inspect

import pytest

from dataforge.evaluators.base import BaseEvaluator
from dataforge.evaluators.llm_judge import LLMJudge
from dataforge.schema import DataRecord


class MockLLM:
    def __init__(self, response: str):
        self._response = response

    async def generate(self, prompt, **kwargs):
        return self._response


def make_record(text: str = "test answer") -> DataRecord:
    r = DataRecord(seed_data={"instruction": "question"})
    r.synthetic_data = {"response": text}
    return r


async def test_llm_judge_passes_above_threshold():
    judge = LLMJudge(llm=MockLLM("4.5"), threshold=4.0)
    record = make_record()
    assert await judge.evaluate(record) is True
    assert record.score == pytest.approx(4.5)


async def test_llm_judge_rejects_below_threshold():
    judge = LLMJudge(llm=MockLLM("3.0"), threshold=4.0)
    record = make_record()
    assert await judge.evaluate(record) is False
    assert record.score == pytest.approx(3.0)


async def test_llm_judge_passes_at_exact_threshold():
    """Boundary: score >= threshold (not strictly greater)."""
    judge = LLMJudge(llm=MockLLM("4.0"), threshold=4.0)
    assert await judge.evaluate(make_record()) is True


async def test_llm_judge_extracts_score_from_text():
    judge = LLMJudge(llm=MockLLM("I give this a score of 4 out of 5."), threshold=4.0)
    assert await judge.evaluate(make_record()) is True


async def test_llm_judge_rejects_on_unparseable_score():
    """No number in response → reject record."""
    judge = LLMJudge(llm=MockLLM("I cannot evaluate this."), threshold=4.0)
    record = make_record()
    assert await judge.evaluate(record) is False
    assert record.score is None


async def test_llm_judge_builtin_criteria_helpfulness():
    judge = LLMJudge(llm=MockLLM("5"), criteria="helpfulness", threshold=4.0)
    assert await judge.evaluate(make_record()) is True


async def test_llm_judge_builtin_criteria_factuality():
    judge = LLMJudge(llm=MockLLM("3"), criteria="factuality", threshold=4.0)
    assert await judge.evaluate(make_record()) is False


async def test_llm_judge_builtin_criteria_logical_reasoning():
    judge = LLMJudge(llm=MockLLM("5"), criteria="logical_reasoning", threshold=4.0)
    assert await judge.evaluate(make_record()) is True


async def test_llm_judge_custom_criteria_string():
    judge = LLMJudge(llm=MockLLM("5"), criteria="Rate clarity:", threshold=4.0)
    assert await judge.evaluate(make_record()) is True


async def test_llm_judge_sets_score_on_record():
    judge = LLMJudge(llm=MockLLM("4.2"), threshold=4.0)
    record = make_record()
    await judge.evaluate(record)
    assert record.score == pytest.approx(4.2)


def test_llm_judge_inherits_base_evaluator():
    assert issubclass(LLMJudge, BaseEvaluator)


def test_llm_judge_evaluate_is_coroutine():
    assert inspect.iscoroutinefunction(LLMJudge.evaluate)


async def test_llm_judge_ignores_large_year_number():
    """Ambiguous text: year 2024 should not be mistaken for score."""
    judge = LLMJudge(llm=MockLLM("2024年，评分3"), threshold=3.0)
    record = make_record()
    assert await judge.evaluate(record) is True
    assert record.score == pytest.approx(3.0)


async def test_llm_judge_out_of_range_score_rejected():
    """Score 6 is out of 1-5 range → None → record rejected."""
    judge = LLMJudge(llm=MockLLM("6"), threshold=4.0)
    record = make_record()
    assert await judge.evaluate(record) is False
    assert record.score is None
