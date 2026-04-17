import pytest
from pydantic import ValidationError

from dataforge.config.schema import ForgeConfig

VALID_CONFIG = {
    "name": "test-dataset",
    "source": {"type": "jsonl", "path": "./seeds.jsonl"},
    "pipeline": [
        {
            "step": "generate",
            "strategy": "evol-instruct",
            "depth": 2,
            "llm": {"provider": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct"},
        }
    ],
    "sink": {"path": "./output.jsonl"},
}


def test_valid_config_parses():
    config = ForgeConfig.model_validate(VALID_CONFIG)
    assert config.name == "test-dataset"
    assert len(config.pipeline) == 1
    assert config.sink.checkpoint_dir == "./.dataforge_runs"  # default


def test_missing_name_raises():
    data = {k: v for k, v in VALID_CONFIG.items() if k != "name"}
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_invalid_provider_raises():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [
        {
            "step": "generate",
            "strategy": "evol-instruct",
            "llm": {"provider": "bad-provider", "model": "test"},
        }
    ]
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_evaluate_step_defaults():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [
        data["pipeline"][0],
        {"step": "evaluate", "evaluator": "regex-filter"},
    ]
    config = ForgeConfig.model_validate(data)
    eval_step = config.pipeline[1]
    assert eval_step.blacklist_patterns == []
    assert eval_step.threshold == 4.0


def test_invalid_step_type_raises():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [{"step": "transform", "llm": {"provider": "vllm", "model": "x"}}]
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_llm_judge_without_llm_raises():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [
        data["pipeline"][0],
        {"step": "evaluate", "evaluator": "llm-judge"},  # llm omitted
    ]
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_generate_depth_zero_raises():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [
        {
            "step": "generate",
            "strategy": "evol-instruct",
            "depth": 0,
            "llm": {"provider": "vllm", "model": "test"},
        }
    ]
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_threshold_out_of_range_raises():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [
        data["pipeline"][0],
        {"step": "evaluate", "evaluator": "regex-filter", "threshold": 10.0},
    ]
    with pytest.raises(ValidationError):
        ForgeConfig.model_validate(data)


def test_invalid_regex_pattern_raises():
    data = dict(VALID_CONFIG)
    data["pipeline"] = [
        data["pipeline"][0],
        {
            "step": "evaluate",
            "evaluator": "regex-filter",
            "blacklist_patterns": ["[unclosed"],
        },
    ]
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        ForgeConfig.model_validate(data)
