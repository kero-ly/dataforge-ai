# tests/test_self_play.py
"""Tests for the Self-Play multi-agent strategy."""
import pytest

from dataforge.registry import get_strategy
from dataforge.schema import DataRecord
from dataforge.strategies.self_play import SelfPlay


class MockLLM:
    def __init__(self, responses=None):
        self.calls = []
        self._responses = responses or ["Mock response"]
        self._idx = 0

    async def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return resp


def make_record(topic="The future of AI"):
    return DataRecord(seed_data={"topic": topic})


async def test_basic_self_play():
    llm = MockLLM(["Opening statement", "Response B", "Follow up A", "Final B"])
    strategy = SelfPlay(llm=llm, turns=2)
    record = make_record()
    result = await strategy.apply(record)

    assert result.synthetic_data is not None
    assert result.synthetic_data["topic"] == "The future of AI"
    conv = result.synthetic_data["conversation"]
    assert len(conv) >= 3  # opener + turns*2 - 1
    assert conv[0]["role"] == "Interviewer"
    assert conv[1]["role"] == "Interviewee"


async def test_single_turn():
    llm = MockLLM(["I'll start!", "Great point!"])
    strategy = SelfPlay(llm=llm, turns=1)
    record = make_record()
    result = await strategy.apply(record)

    conv = result.synthetic_data["conversation"]
    # turns=1: opener (A) + response (B) = 2 messages
    assert len(conv) == 2
    assert conv[0]["role"] == "Interviewer"
    assert conv[1]["role"] == "Interviewee"


async def test_custom_roles():
    llm = MockLLM(["Debate start", "Counterpoint", "Rebuttal", "Final word"])
    strategy = SelfPlay(
        llm=llm, turns=2,
        role_a="Proponent", role_b="Opponent",
    )
    record = make_record()
    result = await strategy.apply(record)

    assert result.synthetic_data["role_a"] == "Proponent"
    assert result.synthetic_data["role_b"] == "Opponent"
    conv = result.synthetic_data["conversation"]
    roles = {msg["role"] for msg in conv}
    assert roles == {"Proponent", "Opponent"}


async def test_two_different_llms():
    llm_a = MockLLM(["A speaks first", "A follows up"])
    llm_b = MockLLM(["B responds"])
    strategy = SelfPlay(llm=llm_a, llm_b=llm_b, turns=1)
    record = make_record()
    result = await strategy.apply(record)

    conv = result.synthetic_data["conversation"]
    assert conv[0]["content"] == "A speaks first"
    assert conv[1]["content"] == "B responds"
    assert len(llm_a.calls) == 1  # opener only
    assert len(llm_b.calls) == 1  # one response


async def test_custom_source_field():
    llm = MockLLM(["Start", "Reply"])
    strategy = SelfPlay(llm=llm, turns=1, source_field="subject")
    record = DataRecord(seed_data={"subject": "Climate change"})
    result = await strategy.apply(record)

    assert result.synthetic_data["topic"] == "Climate change"


async def test_missing_source_field():
    llm = MockLLM()
    strategy = SelfPlay(llm=llm, source_field="missing_field")
    record = DataRecord(seed_data={"topic": "test"})

    with pytest.raises(ValueError, match="missing required field"):
        await strategy.apply(record)


async def test_custom_system_prompts():
    llm = MockLLM(["Custom A", "Custom B"])
    strategy = SelfPlay(
        llm=llm, turns=1,
        system_prompt_a="You are a teacher.",
        system_prompt_b="You are a student.",
    )
    record = make_record()
    await strategy.apply(record)
    assert len(llm.calls) == 2


def test_invalid_turns():
    with pytest.raises(ValueError, match="turns must be >= 1"):
        SelfPlay(llm=MockLLM(), turns=0)


def test_registry_registration():
    cls = get_strategy("self-play")
    assert cls is SelfPlay


async def test_message_building():
    """Verify that _build_messages produces correct chat format."""
    messages = SelfPlay._build_messages(
        system_prompt="Be helpful",
        conversation=[
            {"role": "A", "content": "Hello"},
            {"role": "B", "content": "Hi"},
        ],
        current_role="A",
    )
    assert messages[0] == {"role": "system", "content": "Be helpful"}
    assert messages[1] == {"role": "assistant", "content": "Hello"}  # A is current
    assert messages[2] == {"role": "user", "content": "Hi"}  # B is other


async def test_num_turns_in_output():
    llm = MockLLM(["a", "b", "c", "d", "e"])
    strategy = SelfPlay(llm=llm, turns=2)
    record = make_record()
    result = await strategy.apply(record)

    assert result.synthetic_data["num_turns"] == len(result.synthetic_data["conversation"])
