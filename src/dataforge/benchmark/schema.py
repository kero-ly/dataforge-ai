from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    id: str
    category: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkCaseResult(BaseModel):
    case_id: str
    category: str
    prompt: str
    response: str
    raw_score: float | None = None
    normalized_score: float | None = None
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float | None = None


class BenchmarkTaskSummary(BaseModel):
    task: str
    category_scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float | None = None
    success_rate: float
    num_cases: int
    num_errors: int


class BenchmarkRunSummary(BaseModel):
    benchmark: str
    candidate_name: str
    task_summaries: list[BenchmarkTaskSummary]
    overall_score: float | None = None
    weighted_scores: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
