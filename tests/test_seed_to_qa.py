# tests/test_seed_to_qa.py
"""Tests for the Seed-to-QA strategy."""
import json

import pytest

from dataforge.registry import get_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.seed_to_qa import SeedToQA


class MockLLM:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or ['[{"question": "Q?", "answer": "A.", "difficulty": "easy"}]']
        self._call_idx = 0

    async def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        resp = self.responses[self._call_idx % len(self.responses)]
        self._call_idx += 1
        return resp


def make_record(passage="Python is a high-level programming language."):
    return DataRecord(seed_data={"passage": passage})


async def test_basic_qa_generation():
    qa_data = [
        {"question": "What is Python?", "answer": "A programming language.", "difficulty": "easy"},
    ]
    llm = MockLLM([json.dumps(qa_data)])
    strategy = SeedToQA(llm=llm, qa_per_passage=1)
    record = make_record()
    result = await strategy.apply(record)

    assert result.synthetic_data is not None
    assert result.synthetic_data["passage"] == "Python is a high-level programming language."
    assert len(result.synthetic_data["qa_pairs"]) == 1
    assert result.synthetic_data["qa_pairs"][0]["question"] == "What is Python?"


async def test_multiple_qa_pairs():
    qa_data = [
        {"question": "Q1?", "answer": "A1.", "difficulty": "easy"},
        {"question": "Q2?", "answer": "A2.", "difficulty": "medium"},
        {"question": "Q3?", "answer": "A3.", "difficulty": "hard"},
    ]
    llm = MockLLM([json.dumps(qa_data)])
    strategy = SeedToQA(llm=llm, qa_per_passage=3)
    record = make_record()
    result = await strategy.apply(record)

    assert len(result.synthetic_data["qa_pairs"]) == 3


async def test_custom_source_field():
    qa_data = [{"question": "Q?", "answer": "A.", "difficulty": "easy"}]
    llm = MockLLM([json.dumps(qa_data)])
    strategy = SeedToQA(llm=llm, source_field="text")
    record = DataRecord(seed_data={"text": "Some content"})
    result = await strategy.apply(record)

    assert result.synthetic_data["passage"] == "Some content"


async def test_missing_source_field_raises():
    llm = MockLLM()
    strategy = SeedToQA(llm=llm, source_field="missing")
    record = DataRecord(seed_data={"passage": "test"})

    with pytest.raises(ValueError, match="missing required field"):
        await strategy.apply(record)


async def test_json_with_markdown_fences():
    qa_data = [{"question": "Q?", "answer": "A.", "difficulty": "easy"}]
    fenced = f"```json\n{json.dumps(qa_data)}\n```"
    llm = MockLLM([fenced])
    strategy = SeedToQA(llm=llm, qa_per_passage=1)
    record = make_record()
    result = await strategy.apply(record)

    assert len(result.synthetic_data["qa_pairs"]) == 1


async def test_json_repair_on_failure():
    """First response is bad JSON, repair response is valid."""
    qa_data = [{"question": "Q?", "answer": "A.", "difficulty": "easy"}]
    llm = MockLLM([
        "not valid json",
        json.dumps(qa_data),
    ])
    strategy = SeedToQA(llm=llm, qa_per_passage=1, max_repair_attempts=2)
    record = make_record()
    result = await strategy.apply(record)

    assert len(result.synthetic_data["qa_pairs"]) == 1
    assert len(llm.calls) == 2  # original + repair


async def test_json_repair_exhausted():
    llm = MockLLM(["not json at all"])
    strategy = SeedToQA(llm=llm, qa_per_passage=1, max_repair_attempts=1)
    record = make_record()

    with pytest.raises(ValueError, match="QA JSON self-repair exhausted"):
        await strategy.apply(record)


def test_invalid_qa_per_passage():
    with pytest.raises(ValueError, match="qa_per_passage must be >= 1"):
        SeedToQA(llm=MockLLM(), qa_per_passage=0)


def test_registry_registration():
    cls = get_strategy("seed-to-qa")
    assert cls is SeedToQA


def test_custom_difficulty_levels():
    strategy = SeedToQA(llm=MockLLM(), difficulty_levels=["beginner", "advanced"])
    assert strategy.difficulty_levels == ["beginner", "advanced"]
