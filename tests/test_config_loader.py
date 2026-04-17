import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from dataforge.clients.bailian_client import BailianClient
from dataforge.clients.openai_client import OpenAIClient
from dataforge.clients.vllm_client import vLLMClient
from dataforge.clients.vllm_cluster_client import vLLMClusterClient
from dataforge.config.loader import build_client, build_pipeline, load_config
from dataforge.config.schema import ForgeConfig
from dataforge.pipeline import Pipeline

MINIMAL_CONFIG = {
    "name": "test",
    "source": {"type": "jsonl", "path": "./seeds.jsonl"},
    "pipeline": [
        {
            "step": "generate",
            "strategy": "evol-instruct",
            "depth": 1,
            "llm": {
                "provider": "vllm",
                "model": "test-model",
                "concurrency": 10,
            },
        }
    ],
    "sink": {"path": "./output.jsonl", "checkpoint_dir": "./ckpt"},
}


def test_build_client_vllm():
    from dataforge.config.schema import LLMConfig
    cfg = LLMConfig(provider="vllm", model="test-model")
    client = build_client(cfg)
    assert isinstance(client, vLLMClient)


def test_build_client_vllm_cluster_from_comma_separated_base_url():
    from dataforge.config.schema import LLMConfig

    cfg = LLMConfig(
        provider="vllm",
        model="test-model",
        base_url="http://localhost:8000/v1,http://localhost:8001/v1",
    )
    client = build_client(cfg)

    assert isinstance(client, vLLMClusterClient)
    assert client.base_urls == ["http://localhost:8000/v1", "http://localhost:8001/v1"]


def test_build_client_openai_reads_env_key():
    from dataforge.config.schema import LLMConfig
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
        client = build_client(cfg)
    assert isinstance(client, OpenAIClient)
    assert client.api_key == "test-key"


def test_build_client_openai_explicit_key():
    from dataforge.config.schema import LLMConfig
    cfg = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="explicit-key")
    client = build_client(cfg)
    assert isinstance(client, OpenAIClient)
    assert client.api_key == "explicit-key"


def test_build_client_bailian_reads_env_key_and_default_base_url():
    from dataforge.config.schema import LLMConfig

    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dashscope-key"}):
        cfg = LLMConfig(provider="bailian", model="qwen-plus")
        client = build_client(cfg)

    assert isinstance(client, BailianClient)
    assert client.api_key == "dashscope-key"
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_build_client_bailian_respects_custom_base_url():
    from dataforge.config.schema import LLMConfig

    cfg = LLMConfig(
        provider="bailian",
        model="qwen-plus",
        api_key="dashscope-key",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    client = build_client(cfg)

    assert isinstance(client, BailianClient)
    assert client.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_build_pipeline_returns_correct_types():
    config = ForgeConfig.model_validate(MINIMAL_CONFIG)
    pipeline, input_path, output_path, concurrency = build_pipeline(config)
    assert isinstance(pipeline, Pipeline)
    assert input_path == "./seeds.jsonl"
    assert output_path == "./output.jsonl"
    assert concurrency == 10


def test_build_pipeline_no_generate_step_raises():
    data = dict(MINIMAL_CONFIG)
    data["pipeline"] = [{"step": "evaluate", "evaluator": "regex-filter"}]
    config = ForgeConfig.model_validate(data)
    with pytest.raises(ValueError, match="at least one 'generate' step"):
        build_pipeline(config)


def test_build_pipeline_llm_judge_without_llm_raises():
    data = dict(MINIMAL_CONFIG)
    data["pipeline"] = [
        data["pipeline"][0],
        {"step": "evaluate", "evaluator": "llm-judge"},
    ]
    # This should now raise at schema validation level (model_validator in EvaluateStepConfig)
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_build_pipeline_multiple_generate_steps_raises():
    data = dict(MINIMAL_CONFIG)
    data["pipeline"] = [
        data["pipeline"][0],
        data["pipeline"][0],  # duplicate generate step
    ]
    config = ForgeConfig.model_validate(data)
    with pytest.raises(ValueError, match="exactly one 'generate' step"):
        build_pipeline(config)


def test_load_config_from_yaml_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(MINIMAL_CONFIG, f)
        tmp_path = f.name
    try:
        config = load_config(tmp_path)
        assert config.name == "test"
        assert config.sink.checkpoint_dir == "./ckpt"
    finally:
        os.unlink(tmp_path)


def test_build_client_openai_missing_key_raises():
    from dataforge.config.schema import LLMConfig
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
        with pytest.raises(ValueError, match="API key required"):
            build_client(cfg)


def test_build_client_bailian_missing_key_raises():
    from dataforge.config.schema import LLMConfig

    env = {k: v for k, v in os.environ.items() if k != "DASHSCOPE_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        cfg = LLMConfig(provider="bailian", model="qwen-plus")
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            build_client(cfg)
