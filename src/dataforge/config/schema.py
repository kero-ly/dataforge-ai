from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class LLMConfig(BaseModel):
    provider: Literal["openai", "vllm", "anthropic", "bailian"]
    model: str
    base_url: str | None = None
    api_key: str | None = None
    rpm_limit: int = 60
    tpm_limit: int = 100_000
    concurrency: int = 50
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)


class SourceConfig(BaseModel):
    type: Literal["jsonl", "csv", "parquet"] = "jsonl"
    path: str


class SinkConfig(BaseModel):
    path: str
    format: Literal["jsonl", "csv", "parquet"] = "jsonl"
    checkpoint_dir: str = "./.dataforge_runs"
    checkpoint_backend: Literal["jsonl", "sqlite"] = "jsonl"
    checkpoint_disabled: bool = False
    dead_letter_path: str | None = None
    flush_batch_size: int = 10
    flush_timeout: float = 0.5


class GenerateStepConfig(BaseModel):
    step: Literal["generate"]
    strategy: str = "evol-instruct"
    depth: int = Field(default=3, ge=1)
    mutation_types: list[str] | None = None
    require_reasoning: bool = False
    require_json: bool = False
    llm: LLMConfig
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluateStepConfig(BaseModel):
    step: Literal["evaluate"]
    evaluator: str
    llm: LLMConfig | None = None
    criteria: str = "helpfulness"
    threshold: float = Field(default=4.0, ge=1.0, le=5.0)
    blacklist_patterns: list[str] = Field(default_factory=list)
    require_json: bool = False
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("blacklist_patterns")
    @classmethod
    def validate_regex_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern {pattern!r}: {e}") from e
        return patterns

    @model_validator(mode="after")
    def llm_required_for_llm_judge(self) -> EvaluateStepConfig:
        if self.evaluator == "llm-judge" and self.llm is None:
            raise ValueError("'llm' config is required when evaluator is 'llm-judge'")
        return self


StepConfig = Annotated[
    Union[GenerateStepConfig, EvaluateStepConfig],  # noqa: UP007
    Field(discriminator="step"),
]


class DistributedConfig(BaseModel):
    """Configuration for distributed pipeline execution."""

    enabled: bool = False
    backend: Literal["redis"] = "redis"
    redis_url: str = "redis://localhost:6379"
    role: Literal["coordinator", "worker"] = "worker"
    shared_rate_limit: bool = False
    queue_name: str = "dataforge:tasks"
    result_channel: str = "dataforge:results"


class ForgeConfig(BaseModel):
    name: str
    source: SourceConfig
    pipeline: list[StepConfig]
    sink: SinkConfig
    concurrency: int = 50
    mode: Literal["streaming", "burst"] = "streaming"
    distributed: DistributedConfig | None = None
