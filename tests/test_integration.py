# tests/test_integration.py
"""Smoke test: verifies public API surface matches api_interface.md spec."""
import pytest

openai = pytest.importorskip("openai", reason="openai package not installed")
from dataforge import DataRecord, Pipeline, RecordStatus  # noqa: E402
from dataforge.clients import BailianClient, BaseLLMClient, OpenAIClient, vLLMClient  # noqa: E402
from dataforge.evaluators import BaseEvaluator  # noqa: E402
from dataforge.strategies import BaseStrategy  # noqa: E402


def test_public_imports():
    assert Pipeline is not None
    assert DataRecord is not None
    assert RecordStatus is not None
    assert BaseLLMClient is not None
    assert BailianClient is not None
    assert OpenAIClient is not None
    assert vLLMClient is not None
    assert BaseStrategy is not None
    assert BaseEvaluator is not None


def test_openai_client_init():
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key="sk-fake",
        rpm_limit=500,
        tpm_limit=200_000,
        generation_kwargs={"temperature": 0.7, "max_tokens": 2048},
    )
    assert client.model == "gpt-4o-mini"
    assert client.rpm_limit == 500
    assert client.generation_kwargs["temperature"] == 0.7


def test_vllm_client_init():
    client = vLLMClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        rpm_limit=1000,
    )
    assert client.model == "Qwen/Qwen2.5-7B-Instruct"
    assert client.rpm_limit == 1000


def test_bailian_client_init():
    client = BailianClient(
        model="qwen-plus",
        api_key="sk-fake",
        generation_kwargs={"temperature": 0.2},
    )
    assert client.model == "qwen-plus"
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert client.generation_kwargs["temperature"] == 0.2


def test_pipeline_init():
    """Verify Pipeline can be constructed with keyword args matching the spec."""
    from dataforge.evaluators.base import BaseEvaluator
    from dataforge.schema import DataRecord
    from dataforge.strategies.base import BaseStrategy

    class StubStrategy(BaseStrategy):
        async def apply(self, record: DataRecord) -> DataRecord:
            return record

    class StubEvaluator(BaseEvaluator):
        async def evaluate(self, record: DataRecord) -> bool:
            return True

    pipeline = Pipeline(
        strategy=StubStrategy(),
        evaluators=[StubEvaluator()],
        checkpoint_dir="./test_checkpoint",
        max_retries=3,
    )
    assert pipeline.max_retries == 3
    assert len(pipeline.evaluators) == 1


def test_data_record_all_status_values():
    """All status enum values from api_interface.md must exist."""
    expected = {"PENDING", "GENERATED", "EVALUATING", "REJECTED", "COMPLETED", "FAILED"}
    actual = {s.value for s in RecordStatus}
    assert expected == actual


def test_hello_world_snippet_imports():
    """The Hello World snippet from api_interface.md must import without error."""
    from dataforge import Pipeline
    from dataforge.clients import BailianClient, OpenAIClient, vLLMClient
    from dataforge.evaluators import BaseEvaluator
    from dataforge.strategies import BaseStrategy
    assert all(
        x is not None
        for x in [Pipeline, BailianClient, vLLMClient, OpenAIClient, BaseStrategy, BaseEvaluator]
    )
