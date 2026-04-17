# tests/test_evol_instruct.py
from unittest.mock import AsyncMock

import pytest

from dataforge.schema import DataRecord
from dataforge.strategies.base import BaseStrategy
from dataforge.strategies.evol_instruct import EvolInstruct


class MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    async def generate(self, prompt, **kwargs):
        resp = self._responses[self._index % len(self._responses)]
        self._index += 1
        return resp


def make_record(instruction: str = "What is gradient descent?") -> DataRecord:
    return DataRecord(seed_data={"instruction": instruction})


async def test_evol_instruct_depth_1():
    llm = MockLLM(["evolved instruction"])
    strategy = EvolInstruct(llm=llm, depth=1, mutation_types=["constraints"])
    result = await strategy.apply(make_record())
    assert result.synthetic_data["instruction"] == "evolved instruction"


async def test_evol_instruct_depth_calls_llm_n_times():
    responses = [f"step{i}" for i in range(3)]
    llm = MockLLM(responses)
    strategy = EvolInstruct(llm=llm, depth=3, mutation_types=["deepen"])
    result = await strategy.apply(make_record())
    assert llm._index == 3
    assert result.synthetic_data["instruction"] == "step2"


async def test_evol_instruct_returns_same_record():
    llm = MockLLM(["evolved"])
    strategy = EvolInstruct(llm=llm, depth=1)
    record = make_record()
    result = await strategy.apply(record)
    assert result is record


async def test_evol_instruct_invalid_mutation_type_raises():
    with pytest.raises(ValueError, match="mutation_types"):
        EvolInstruct(llm=AsyncMock(), depth=1, mutation_types=["invalid_type"])


def test_evol_instruct_default_mutation_types():
    strategy = EvolInstruct(llm=AsyncMock(), depth=1)
    assert len(strategy.mutation_types) > 0
    assert all(m in EvolInstruct.VALID_MUTATION_TYPES for m in strategy.mutation_types)


async def test_evol_instruct_require_reasoning_with_think_tags():
    # depth=1 evolves once, then generates CoT response
    llm = MockLLM(["evolved instruction", "<think>my reasoning</think>final answer"])
    strategy = EvolInstruct(llm=llm, depth=1, require_reasoning=True)
    result = await strategy.apply(make_record())
    assert result.synthetic_data["instruction"] == "evolved instruction"
    assert result.synthetic_data["reasoning"] == "my reasoning"
    assert result.synthetic_data["response"] == "final answer"


async def test_evol_instruct_require_reasoning_no_think_tags():
    """No <think> tags → reasoning="", response=full text."""
    llm = MockLLM(["evolved", "response without tags"])
    strategy = EvolInstruct(llm=llm, depth=1, require_reasoning=True)
    result = await strategy.apply(make_record())
    assert result.synthetic_data["reasoning"] == ""
    assert result.synthetic_data["response"] == "response without tags"


async def test_evol_instruct_require_json_valid_response():
    llm = MockLLM(['{"instruction": "evolved", "difficulty": "hard"}'])
    strategy = EvolInstruct(llm=llm, depth=1, require_json=True, mutation_types=["constraints"])
    result = await strategy.apply(make_record())
    assert isinstance(result.synthetic_data, dict)
    assert result.synthetic_data["instruction"] == "evolved"


async def test_evol_instruct_require_json_strips_markdown_fence():
    """JSON inside ```json fences should parse without a repair call.

    With depth=1 + require_json=True there are 2 LLM calls:
      - call 1: evolution step
      - call 2: _generate_json internal call
    Fence stripping on call 2's output avoids a 3rd (repair) call.
    """
    fence_response = '```json\n{"instruction": "evolved"}\n```'
    llm = MockLLM([fence_response, fence_response])  # evolution + JSON gen
    strategy = EvolInstruct(llm=llm, depth=1, require_json=True, mutation_types=["deepen"])
    result = await strategy.apply(make_record())
    assert result.synthetic_data["instruction"] == "evolved"
    assert llm._index == 2  # evolution call + JSON gen call, no repair needed


async def test_evol_instruct_require_json_self_repair():
    """Invalid JSON triggers one repair call, succeeds on second attempt."""
    llm = MockLLM(["not valid json", '{"instruction": "repaired"}'])
    strategy = EvolInstruct(llm=llm, depth=1, require_json=True, mutation_types=["concretize"])
    result = await strategy.apply(make_record())
    assert result.synthetic_data["instruction"] == "repaired"
    assert llm._index == 2  # Original call + 1 repair


async def test_evol_instruct_require_json_exhausted_raises():
    """All repair attempts fail → ValueError raised."""
    llm = MockLLM(["bad json", "still bad json", "still bad"])
    strategy = EvolInstruct(llm=llm, depth=1, require_json=True, mutation_types=["constraints"])
    with pytest.raises(ValueError, match="repair"):
        await strategy.apply(make_record())


def test_evol_instruct_inherits_base_strategy():
    assert issubclass(EvolInstruct, BaseStrategy)


async def test_evol_instruct_round_robin_schedule():
    """Round-robin schedule cycles through mutation types in order."""
    mutations = ["constraints", "deepen", "concretize"]
    llm = MockLLM([f"evolved{i}" for i in range(9)])
    strategy = EvolInstruct(
        llm=llm, depth=1, mutation_types=mutations, mutation_schedule="round_robin"
    )
    # Apply 6 records — should cycle through mutations twice
    selected = []
    for _ in range(6):
        record = make_record()
        await strategy.apply(record)
        # The counter increments after each call to _next_mutation
        selected.append(strategy._counter - 1)

    # Verify cycling: indices 0,1,2,0,1,2 correspond to mutations in order
    expected_mutations = mutations * 2
    for i, idx in enumerate(selected):
        assert mutations[idx % len(mutations)] == expected_mutations[i]


async def test_evol_instruct_batch_schedule():
    """Batch schedule uses same mutation for batch_size consecutive records."""
    mutations = ["constraints", "deepen", "concretize"]
    batch_size = 3
    llm = MockLLM([f"evolved{i}" for i in range(9)])
    strategy = EvolInstruct(
        llm=llm,
        depth=1,
        mutation_types=mutations,
        mutation_schedule="batch",
        batch_size=batch_size,
    )
    # Apply 9 records with batch_size=3: first 3 → constraints, next 3 → deepen, last 3 → concretize
    results = []
    for _ in range(9):
        await strategy.apply(make_record())
        # Record which mutation was last selected via counter
        results.append(
            mutations[((strategy._counter - 1) // batch_size) % len(mutations)]
        )

    assert results[:3] == ["constraints"] * 3
    assert results[3:6] == ["deepen"] * 3
    assert results[6:9] == ["concretize"] * 3


def test_evol_instruct_random_schedule_is_default():
    """Default schedule is 'random' for backward compatibility."""
    strategy = EvolInstruct(llm=AsyncMock(), depth=1)
    assert strategy.mutation_schedule == "random"


def test_evol_instruct_invalid_schedule_raises():
    """Invalid mutation_schedule raises ValueError."""
    with pytest.raises(ValueError, match="mutation_schedule"):
        EvolInstruct(llm=AsyncMock(), depth=1, mutation_schedule="invalid")
