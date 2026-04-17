# tests/test_multi_criteria.py
"""Tests for the MultiCriteriaEvaluator."""

import pytest

from dataforge.evaluators.multi_criteria import MultiCriteriaEvaluator
from dataforge.registry import get_evaluator
from dataforge.schema import DataRecord


class MockLLM:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or ['{"helpfulness": 4, "accuracy": 5, "safety": 4}']
        self._idx = 0

    async def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp


def make_record():
    return DataRecord(
        seed_data={"instruction": "test"},
        synthetic_data={"response": "test response"},
    )


async def test_basic_evaluation_pass():
    llm = MockLLM(['{"helpfulness": 5, "accuracy": 4, "safety": 5}'])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=3.5)
    record = make_record()
    result = await evaluator.evaluate(record)

    assert result is True
    assert "criteria_scores" in record.metadata
    assert record.score is not None
    assert record.score >= 3.5


async def test_basic_evaluation_fail():
    llm = MockLLM(['{"helpfulness": 2, "accuracy": 1, "safety": 2}'])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=3.5)
    record = make_record()
    result = await evaluator.evaluate(record)

    assert result is False
    assert record.score is not None
    assert record.score < 3.5


async def test_custom_criteria_and_weights():
    llm = MockLLM(['{"relevance": 5, "creativity": 3}'])
    evaluator = MultiCriteriaEvaluator(
        llm=llm,
        criteria={"relevance": 2.0, "creativity": 1.0},
        threshold=4.0,
    )
    record = make_record()
    result = await evaluator.evaluate(record)

    # weighted avg = (5*2/3 + 3*1/3) = 10/3 + 1 = 4.333
    assert result is True
    assert record.score == pytest.approx(4.3333, rel=0.01)


async def test_json_in_markdown_fences():
    llm = MockLLM(['```json\n{"helpfulness": 4, "accuracy": 4, "safety": 4}\n```'])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=3.0)
    record = make_record()
    result = await evaluator.evaluate(record)

    assert result is True


async def test_fallback_parse_unstructured():
    """When JSON parsing fails, fall back to key: value extraction."""
    llm = MockLLM(["Here are the scores:\nhelpfulness: 4\naccuracy: 5\nsafety: 3"])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=3.0)
    record = make_record()
    result = await evaluator.evaluate(record)

    assert result is True
    assert "criteria_scores" in record.metadata


async def test_completely_unparseable():
    llm = MockLLM(["I cannot evaluate this content."])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=3.0)
    record = make_record()
    result = await evaluator.evaluate(record)

    assert result is False


async def test_partial_scores():
    """Missing dimensions are treated as 0."""
    llm = MockLLM(['{"helpfulness": 5}'])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=2.0)
    record = make_record()
    result = await evaluator.evaluate(record)

    # Only helpfulness has score 5, others are 0.
    # weighted = 5*(1/3) + 0 + 0 = 1.667
    assert result is False


async def test_scores_out_of_range_ignored():
    """Scores outside 1-5 are ignored."""
    llm = MockLLM(['{"helpfulness": 10, "accuracy": 4, "safety": 0}'])
    evaluator = MultiCriteriaEvaluator(llm=llm, threshold=1.0)
    record = make_record()
    await evaluator.evaluate(record)
    assert "criteria_scores" in record.metadata
    assert record.metadata["criteria_scores"].get("helpfulness") is None


def test_empty_criteria_raises():
    with pytest.raises(ValueError, match="at least one dimension"):
        MultiCriteriaEvaluator(llm=MockLLM(), criteria={})


def test_registry_registration():
    cls = get_evaluator("multi-criteria")
    assert cls is MultiCriteriaEvaluator
