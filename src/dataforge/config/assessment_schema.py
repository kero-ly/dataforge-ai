from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from dataforge.config.schema import LLMConfig


class AssessmentSourceConfig(BaseModel):
    path: str
    format: Literal["auto", "dataforge_jsonl", "instruction_response_jsonl"] = "auto"


class AssessmentSuiteConfig(BaseModel):
    name: str = "sft_readiness_v1"
    sample_size: int = Field(default=1000, ge=1)
    sample_seed: int = 42


class ReferenceCorpusConfig(BaseModel):
    path: str
    enabled: bool = False
    fuzzy_overlap_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class AssessmentOutputConfig(BaseModel):
    dir: str
    formats: list[Literal["json", "md", "html"]] = Field(
        default_factory=lambda: ["json", "md", "html"]
    )
    persist_record_results: bool = True


class AssessmentConfig(BaseModel):
    kind: Literal["assessment"] = "assessment"
    name: str
    source: AssessmentSourceConfig
    suite: AssessmentSuiteConfig = Field(default_factory=AssessmentSuiteConfig)
    judge_llm: LLMConfig | None = None
    embedding: LLMConfig | None = None
    reference_corpus: ReferenceCorpusConfig | None = None
    output: AssessmentOutputConfig
    max_concurrency: int = Field(default=20, ge=1)
