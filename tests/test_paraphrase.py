# tests/test_paraphrase.py
"""Tests for the Paraphrase strategy."""
import pytest

from dataforge.registry import get_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.paraphrase import Paraphrase


class MockLLM:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or ["Rewritten text here."]
        self._call_idx = 0

    async def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        resp = self.responses[self._call_idx % len(self.responses)]
        self._call_idx += 1
        return resp


def make_record(instruction="Write a function to sort a list"):
    return DataRecord(seed_data={"instruction": instruction})


async def test_single_variant():
    llm = MockLLM(["A rewritten version of the instruction."])
    strategy = Paraphrase(llm=llm, n_variants=1)
    record = make_record()
    result = await strategy.apply(record)

    assert result.synthetic_data is not None
    assert result.synthetic_data["instruction"] == "Write a function to sort a list"
    assert result.synthetic_data["paraphrase"] == "A rewritten version of the instruction."
    assert len(llm.calls) == 1


async def test_multiple_variants():
    llm = MockLLM(["1. First version\n2. Second version\n3. Third version"])
    strategy = Paraphrase(llm=llm, n_variants=3)
    record = make_record()
    result = await strategy.apply(record)

    assert result.synthetic_data is not None
    assert result.synthetic_data["instruction"] == "Write a function to sort a list"
    assert len(result.synthetic_data["variants"]) == 3
    assert result.synthetic_data["variants"][0] == "First version"


async def test_custom_source_field():
    llm = MockLLM(["Paraphrased content."])
    strategy = Paraphrase(llm=llm, source_field="text")
    record = DataRecord(seed_data={"text": "Hello world"})
    result = await strategy.apply(record)

    assert result.synthetic_data["text"] == "Hello world"
    assert result.synthetic_data["paraphrase"] == "Paraphrased content."


async def test_missing_source_field_raises():
    llm = MockLLM()
    strategy = Paraphrase(llm=llm, source_field="missing_field")
    record = DataRecord(seed_data={"instruction": "test"})

    with pytest.raises(ValueError, match="missing required field"):
        await strategy.apply(record)


def test_invalid_n_variants():
    with pytest.raises(ValueError, match="n_variants must be >= 1"):
        Paraphrase(llm=MockLLM(), n_variants=0)


def test_registry_registration():
    cls = get_strategy("paraphrase")
    assert cls is Paraphrase


def test_parse_variants_numbered():
    result = Paraphrase._parse_variants("1. Alpha\n2. Beta\n3. Gamma", 3)
    assert result == ["Alpha", "Beta", "Gamma"]


def test_parse_variants_truncates_extra():
    result = Paraphrase._parse_variants("1. A\n2. B\n3. C\n4. D", 2)
    assert len(result) == 2


def test_parse_variants_without_numbers():
    result = Paraphrase._parse_variants("Alpha\nBeta", 2)
    assert result == ["Alpha", "Beta"]
