from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dataforge.schema import DataRecord


class AssessmentResult(BaseModel):
    evaluator: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    dimension_scores: dict[str, float | None] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    estimated_cost_usd: float | None = None


class RecordAssessment(BaseModel):
    record_id: str
    line_number: int
    source_path: str
    normalized_record: DataRecord
    results: list[AssessmentResult]
    aggregate_score: float | None = None
    passed_all_required: bool
    tags: list[str] = Field(default_factory=list)


class EvaluatorSummary(BaseModel):
    evaluator: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float | None = None
    p50_score: float | None = None
    p95_score: float | None = None
    reason_code_counts: dict[str, int] = Field(default_factory=dict)


class DatasetAssessmentSummary(BaseModel):
    suite: str
    total_records: int
    sampled_records: int
    sample_seed: int
    overall_quality_score: float | None = None
    evaluator_summaries: list[EvaluatorSummary]
    dataset_metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
