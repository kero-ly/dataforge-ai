# src/dataforge/metrics.py
"""Built-in metrics collector and pipeline result reporting.

``MetricsCollector`` is a ``PipelineHook`` that automatically tracks
counts, timing, and scores during pipeline execution.  At the end of
a run it produces a ``PipelineResult`` summary.

Usage::

    metrics = MetricsCollector()
    result = await pipeline.run(..., hooks=[metrics])
    print(result)  # PipelineResult with all stats
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from dataforge.hooks import PipelineHook
from dataforge.schema import DataRecord


@dataclass
class PipelineResult:
    """Summary statistics for a completed pipeline run."""

    total_records: int = 0
    completed: int = 0
    rejected: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    records_per_second: float = 0.0
    scores: list[float] = field(default_factory=list)
    assessment_avg_score: float | None = None
    assessment_pass_rate: float | None = None
    evaluator_breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_score(self) -> float | None:
        """Average evaluator score across completed records, or None if no scores."""
        return sum(self.scores) / len(self.scores) if self.scores else None

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "Pipeline Result",
            "=" * 40,
            f"  Total processed : {self.total_records}",
            f"  Completed       : {self.completed}",
            f"  Rejected        : {self.rejected}",
            f"  Failed          : {self.failed}",
            f"  Elapsed         : {self.elapsed_seconds:.1f}s",
            f"  Throughput      : {self.records_per_second:.2f} rec/s",
        ]
        avg = self.avg_score
        if avg is not None:
            lines.append(f"  Avg score       : {avg:.2f}")
        if self.assessment_avg_score is not None:
            lines.append(f"  Assess avg      : {self.assessment_avg_score:.2f}")
        if self.assessment_pass_rate is not None:
            lines.append(f"  Assess pass     : {self.assessment_pass_rate:.2%}")
        return "\n".join(lines)


class MetricsCollector(PipelineHook):
    """Pipeline hook that collects execution metrics.

    After the pipeline finishes, access ``.result`` for a ``PipelineResult``.
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._completed = 0
        self._rejected = 0
        self._failed = 0
        self._scores: list[float] = []
        self._assessment_scores: list[float] = []
        self._assessment_passes = 0
        self._assessment_total = 0
        self._evaluator_breakdown: dict[str, dict[str, int]] = {}
        self.result: PipelineResult | None = None

    async def on_pipeline_start(self, *, input_path: str, output_path: str, concurrency: int) -> None:
        self._start_time = time.monotonic()

    async def on_record_completed(self, record: DataRecord) -> None:
        self._completed += 1
        if record.score is not None:
            self._scores.append(record.score)
        self._record_assessment(record)

    async def on_record_rejected(self, record: DataRecord) -> None:
        self._rejected += 1
        self._record_assessment(record)

    async def on_record_failed(self, record: DataRecord, error: Exception) -> None:
        self._failed += 1

    async def on_pipeline_end(self) -> None:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        total = self._completed + self._rejected + self._failed
        self.result = PipelineResult(
            total_records=total,
            completed=self._completed,
            rejected=self._rejected,
            failed=self._failed,
            elapsed_seconds=elapsed,
            records_per_second=total / elapsed if elapsed > 0 else 0.0,
            scores=list(self._scores),
            assessment_avg_score=(
                sum(self._assessment_scores) / len(self._assessment_scores)
                if self._assessment_scores
                else None
            ),
            assessment_pass_rate=(
                self._assessment_passes / self._assessment_total
                if self._assessment_total
                else None
            ),
            evaluator_breakdown=json.loads(json.dumps(self._evaluator_breakdown)),
        )

    def _record_assessment(self, record: DataRecord) -> None:
        payload = record.metadata.get("assessment")
        if not isinstance(payload, dict):
            return
        aggregate = payload.get("aggregate_score")
        if isinstance(aggregate, (int, float)):
            self._assessment_scores.append(float(aggregate))
        passed_all = payload.get("passed_all")
        if isinstance(passed_all, bool):
            self._assessment_total += 1
            self._assessment_passes += int(passed_all)
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            name = str(result.get("evaluator") or "unknown")
            row = self._evaluator_breakdown.setdefault(name, {"passed": 0, "failed": 0})
            row["passed" if result.get("passed") else "failed"] += 1
