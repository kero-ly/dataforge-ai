from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataforge.config.schema import LLMConfig


class BenchmarkTaskConfig(BaseModel):
    name: str
    weight: float | None = Field(default=None, gt=0.0)


class BenchmarkOutputConfig(BaseModel):
    dir: str
    formats: list[Literal["json", "md", "html"]] = Field(
        default_factory=lambda: ["json", "md", "html"]
    )


class BenchmarkConfig(BaseModel):
    kind: Literal["benchmark"] = "benchmark"
    name: str
    candidate: LLMConfig
    judge_llm: LLMConfig | None = None
    tasks: list[BenchmarkTaskConfig]
    output: BenchmarkOutputConfig
