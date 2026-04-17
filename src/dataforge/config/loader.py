# src/dataforge/config/loader.py
from __future__ import annotations

import os
import re

import yaml

from dataforge.assessment.runner import AssessmentRunner
from dataforge.benchmark.runner import BenchmarkRunner
from dataforge.clients import BailianClient, OpenAIClient, vLLMClient, vLLMClusterClient
from dataforge.clients.base import BaseLLMClient
from dataforge.config.assessment_schema import AssessmentConfig
from dataforge.config.benchmark_schema import BenchmarkConfig
from dataforge.config.schema import (
    EvaluateStepConfig,
    ForgeConfig,
    GenerateStepConfig,
    LLMConfig,
)
from dataforge.evaluators.base import BaseEvaluator
from dataforge.pipeline import Pipeline
from dataforge.registry import get_evaluator, get_strategy
from dataforge.strategies.base import BaseStrategy

_ENV_KEYS: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "bailian": "DASHSCOPE_API_KEY",
    "vllm": None,
}


def _expand_env_placeholders(value: object) -> object:
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(v) for v in value]
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")
        return pattern.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    return value


def load_config(path: str) -> ForgeConfig:
    """Read a YAML file and return a validated ForgeConfig."""
    with open(path, encoding="utf-8") as f:
        data = _expand_env_placeholders(yaml.safe_load(f))
    return ForgeConfig.model_validate(data)


def load_assessment_config(path: str) -> AssessmentConfig:
    with open(path, encoding="utf-8") as f:
        data = _expand_env_placeholders(yaml.safe_load(f))
    return AssessmentConfig.model_validate(data)


def load_benchmark_config(path: str) -> BenchmarkConfig:
    with open(path, encoding="utf-8") as f:
        data = _expand_env_placeholders(yaml.safe_load(f))
    return BenchmarkConfig.model_validate(data)


def build_client(cfg: LLMConfig) -> BaseLLMClient:
    """Instantiate the correct LLM client from a LLMConfig."""
    api_key = cfg.api_key
    if api_key is None:
        env_var = _ENV_KEYS.get(cfg.provider)
        if env_var:
            api_key = os.environ.get(env_var)

    if cfg.provider in ("openai", "anthropic", "bailian") and not api_key:
        env_var = _ENV_KEYS.get(cfg.provider, "")
        raise ValueError(
            f"API key required for provider '{cfg.provider}'. "
            f"Set 'api_key' in the config or export {env_var}."
        )

    if cfg.provider == "vllm":
        if cfg.base_url and "," in cfg.base_url:
            return vLLMClusterClient(
                model=cfg.model,
                base_urls=cfg.base_url.split(","),
                rpm_limit=cfg.rpm_limit,
                tpm_limit=cfg.tpm_limit,
                generation_kwargs=cfg.generation_kwargs,
            )
        return vLLMClient(
            model=cfg.model,
            base_url=cfg.base_url or "http://localhost:8000/v1",
            rpm_limit=cfg.rpm_limit,
            tpm_limit=cfg.tpm_limit,
            generation_kwargs=cfg.generation_kwargs,
        )
    if cfg.provider == "anthropic" and cfg.base_url is None:
        import warnings
        warnings.warn(
            "provider='anthropic' uses the OpenAI-compatible SDK. "
            "Set 'base_url: https://api.anthropic.com/v1' in your config "
            "to target the Anthropic API directly.",
            UserWarning,
            stacklevel=2,
        )
    if cfg.provider == "bailian":
        return BailianClient(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            rpm_limit=cfg.rpm_limit,
            tpm_limit=cfg.tpm_limit,
            generation_kwargs=cfg.generation_kwargs,
        )
    # "openai" and "anthropic" both use the OpenAI-compatible SDK interface
    return OpenAIClient(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        rpm_limit=cfg.rpm_limit,
        tpm_limit=cfg.tpm_limit,
        generation_kwargs=cfg.generation_kwargs,
    )


def build_strategy(step: GenerateStepConfig) -> BaseStrategy:
    """Build a strategy from a GenerateStepConfig using the plugin registry."""
    # Ensure built-in strategies are registered
    import dataforge.strategies  # noqa: F401

    llm = build_client(step.llm)
    strategy_cls = get_strategy(step.strategy)

    if step.strategy == "evol-instruct":
        # Backward compat: use dedicated fields for EvolInstruct
        return strategy_cls(
            llm=llm,
            depth=step.depth,
            mutation_types=step.mutation_types,
            require_reasoning=step.require_reasoning,
            require_json=step.require_json,
        )
    # Generic strategies: pass llm + params dict
    return strategy_cls(llm=llm, **step.params)


def build_evaluator(step: EvaluateStepConfig) -> BaseEvaluator:
    """Build an evaluator from an EvaluateStepConfig using the plugin registry."""
    # Ensure built-in evaluators are registered
    import dataforge.dedup  # noqa: F401
    import dataforge.evaluators  # noqa: F401

    evaluator_cls = get_evaluator(step.evaluator)

    # Build kwargs from well-known fields
    kwargs: dict[str, object] = {}
    if step.llm is not None:
        kwargs["llm"] = build_client(step.llm)
    if step.criteria != "helpfulness":
        kwargs["criteria"] = step.criteria
    if step.threshold != 4.0:
        kwargs["threshold"] = step.threshold
    if step.blacklist_patterns:
        kwargs["blacklist_patterns"] = step.blacklist_patterns
    if step.require_json:
        kwargs["require_json"] = step.require_json
    # Merge generic params (params take precedence)
    kwargs.update(step.params)

    return evaluator_cls(**kwargs)


def build_pipeline(config: ForgeConfig) -> tuple[Pipeline, str, str, int]:
    """Assemble a Pipeline from a ForgeConfig.

    For non-JSONL source formats (csv, parquet), the input is converted to
    a temporary JSONL file before pipeline execution.

    Returns:
        (pipeline, input_path, output_path, concurrency)
    """
    strategy: BaseStrategy | None = None
    evaluators: list[BaseEvaluator] = []
    concurrency = 50

    for step in config.pipeline:
        if step.step == "generate":
            if strategy is not None:
                raise ValueError(
                    "Pipeline config must contain exactly one 'generate' step"
                )
            strategy = build_strategy(step)
            concurrency = step.llm.concurrency
        elif step.step == "evaluate":
            evaluators.append(build_evaluator(step))

    if strategy is None:
        raise ValueError(
            "Pipeline config must contain at least one 'generate' step"
        )

    input_path = config.source.path

    # Convert non-JSONL sources to a temp JSONL file for the pipeline
    if config.source.type != "jsonl":
        from dataforge.io import read_records, write_jsonl_records

        records = read_records(input_path, config.source.type)
        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
        import os

        os.close(tmp_fd)
        write_jsonl_records(records, tmp_path)
        input_path = tmp_path

    pipeline = Pipeline(
        strategy=strategy,
        evaluators=evaluators,
        checkpoint_dir=config.sink.checkpoint_dir,
        dead_letter_path=config.sink.dead_letter_path,
        checkpoint_backend=config.sink.checkpoint_backend,
        checkpoint_disabled=config.sink.checkpoint_disabled,
        flush_batch_size=config.sink.flush_batch_size,
        flush_timeout=config.sink.flush_timeout,
    )
    return pipeline, input_path, config.sink.path, concurrency


def build_assessment_runner(config: AssessmentConfig) -> AssessmentRunner:
    runtime_config = config.model_copy(deep=True)
    if runtime_config.judge_llm is not None:
        object.__setattr__(runtime_config, "_judge_client", build_client(runtime_config.judge_llm))
    return AssessmentRunner(runtime_config)


def build_benchmark_runner(config: BenchmarkConfig) -> BenchmarkRunner:
    runtime_config = config.model_copy(deep=True)
    object.__setattr__(runtime_config, "_candidate_client", build_client(runtime_config.candidate))
    if runtime_config.judge_llm is not None:
        object.__setattr__(runtime_config, "_judge_client", build_client(runtime_config.judge_llm))
    return BenchmarkRunner(runtime_config)
