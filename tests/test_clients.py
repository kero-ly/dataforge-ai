from __future__ import annotations

# tests/test_clients.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dataforge.clients.bailian_client import BailianClient
from dataforge.clients.base import BaseLLMClient
from dataforge.clients.openai_client import OpenAIClient
from dataforge.clients.vllm_client import vLLMClient
from dataforge.clients.vllm_cluster_client import vLLMClusterClient


class MockClient(BaseLLMClient):
    async def generate(self, prompt: str | list[dict], **kwargs) -> str:
        return "mock response"


class RecordingObserver:
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.errors = 0
        self.last_total_tokens = None

    def on_request_start(self, **kwargs) -> None:
        self.starts += 1

    def on_request_end(self, **kwargs) -> None:
        self.ends += 1
        self.last_total_tokens = kwargs.get("total_tokens")

    def on_request_error(self, **kwargs) -> None:
        self.errors += 1


async def test_base_client_generate():
    client = MockClient(model="mock-model")
    result = await client.generate("Hello")
    assert result == "mock response"


def test_base_client_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        BaseLLMClient(model="test")


async def test_openai_client_generate_returns_string():
    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-fake")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]

    with patch.object(client._aclient.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
        result = await client.generate("Say hello")
    assert result == "Hello!"


async def test_openai_client_wraps_string_prompt_in_messages():
    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-fake")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hi"))]

    captured_kwargs: dict = {}

    async def mock_create(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    with patch.object(client._aclient.chat.completions, "create", side_effect=mock_create):
        await client.generate("test prompt")

    messages = captured_kwargs["messages"]
    assert messages == [{"role": "user", "content": "test prompt"}]


async def test_openai_client_passes_list_prompt_directly():
    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-fake")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="reply"))]

    captured_kwargs: dict = {}

    async def mock_create(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hello"},
    ]
    with patch.object(client._aclient.chat.completions, "create", side_effect=mock_create):
        await client.generate(messages)

    assert captured_kwargs["messages"] == messages


def test_openai_client_stores_generation_kwargs():
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key="sk-fake",
        generation_kwargs={"temperature": 0.7, "max_tokens": 512},
    )
    assert client.generation_kwargs["temperature"] == 0.7
    assert client.generation_kwargs["max_tokens"] == 512


def test_openai_client_has_default_request_timeout():
    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-fake")
    assert client.request_timeout == 120.0


def test_bailian_client_default_base_url():
    client = BailianClient(model="qwen-plus", api_key="sk-fake")
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_bailian_client_custom_base_url():
    client = BailianClient(
        model="qwen-plus",
        api_key="sk-fake",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    )
    assert client.base_url == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"


async def test_openai_client_call_time_kwargs_override_generation_kwargs():
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key="sk-fake",
        generation_kwargs={"temperature": 0.9, "max_tokens": 100},
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

    captured_kwargs: dict = {}

    async def mock_create(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_response

    with patch.object(client._aclient.chat.completions, "create", side_effect=mock_create):
        await client.generate("hi", temperature=0.1)  # call-time should win

    assert captured_kwargs["temperature"] == 0.1   # call-time value, not 0.9
    assert captured_kwargs["max_tokens"] == 100     # generation_kwargs value preserved


async def test_openai_client_notifies_request_observer():
    client = OpenAIClient(model="gpt-4o-mini", api_key="sk-fake")
    observer = RecordingObserver()
    client.add_observer(observer)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
    mock_response.usage = MagicMock(prompt_tokens=12, completion_tokens=5, total_tokens=17)

    with patch.object(client._aclient.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
        result = await client.generate("Say hello")

    assert result == "Hello!"
    assert observer.starts == 1
    assert observer.ends == 1
    assert observer.errors == 0
    assert observer.last_total_tokens == 17


async def test_bailian_client_generate_returns_string():
    client = BailianClient(model="qwen-plus", api_key="sk-fake")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Bailian response"))]

    with patch.object(
        client._aclient.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ):
        result = await client.generate("hello")

    assert result == "Bailian response"


def test_vllm_client_default_base_url():
    client = vLLMClient(model="Qwen/Qwen2.5-7B-Instruct")
    assert client.base_url == "http://localhost:8000/v1"


def test_vllm_client_custom_base_url():
    client = vLLMClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://192.168.1.10:8000/v1",
    )
    assert client.base_url == "http://192.168.1.10:8000/v1"


def test_vllm_client_high_default_rpm():
    """vLLM runs locally, so default RPM limit should be much higher than cloud APIs."""
    client = vLLMClient(model="test-model")
    assert client.rpm_limit >= 500


async def test_vllm_client_generate_returns_string():
    client = vLLMClient(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="vLLM response"))]

    with patch.object(client._aclient.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
        result = await client.generate("hello")
    assert result == "vLLM response"


async def test_vllm_cluster_client_round_robins_across_endpoints():
    client = vLLMClusterClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_urls=["http://localhost:8000/v1", "http://localhost:8001/v1"],
    )
    client._clients[0].generate = AsyncMock(return_value="first")
    client._clients[1].generate = AsyncMock(return_value="second")

    assert await client.generate("hello") == "first"
    assert await client.generate("hello") == "second"
    assert await client.generate("hello") == "first"


def test_vllm_cluster_client_propagates_observers():
    client = vLLMClusterClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_urls=["http://localhost:8000/v1", "http://localhost:8001/v1"],
    )
    observer = RecordingObserver()

    client.add_observer(observer)

    assert observer in client._request_observers
    assert all(observer in subclient._request_observers for subclient in client._clients)


async def test_vllm_cluster_client_prefix_affinity_uses_limited_replica_set():
    client = vLLMClusterClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_urls=[
            "http://localhost:8000/v1",
            "http://localhost:8001/v1",
            "http://localhost:8002/v1",
            "http://localhost:8003/v1",
        ],
        routing_strategy="prefix_affinity",
        prefix_replication=2,
    )
    for idx, subclient in enumerate(client._clients):
        subclient.generate = AsyncMock(return_value=f"endpoint-{idx}")

    seen = {await client.generate("Rewrite the following instruction to add constraints:\n\nx") for _ in range(8)}

    assert len(seen) <= 2


async def test_vllm_cluster_client_generate_raw_respects_prefix_affinity():
    client = vLLMClusterClient(
        model="Qwen/Qwen2.5-7B-Instruct",
        base_urls=[
            "http://localhost:8000/v1",
            "http://localhost:8001/v1",
            "http://localhost:8002/v1",
            "http://localhost:8003/v1",
        ],
        routing_strategy="prefix_affinity",
        prefix_replication=2,
    )
    for idx, subclient in enumerate(client._clients):
        subclient.generate_raw = AsyncMock(return_value=f"endpoint-{idx}")

    messages = [{"role": "user", "content": "Shared prefix request payload"}]
    seen = {await client.generate_raw(messages) for _ in range(8)}

    assert len(seen) <= 2
