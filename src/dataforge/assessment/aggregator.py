from __future__ import annotations

from collections import Counter

from dataforge.assessment.schema import (
    DatasetAssessmentSummary,
    EvaluatorSummary,
    RecordAssessment,
)
from dataforge.assessment.utils import percentile


def summarize_evaluators(records: list[RecordAssessment]) -> list[EvaluatorSummary]:
    grouped: dict[str, list[tuple[bool, float | None, list[str]]]] = {}
    for record in records:
        for result in record.results:
            grouped.setdefault(result.evaluator, []).append(
                (result.passed, result.score, result.reason_codes)
            )

    summaries: list[EvaluatorSummary] = []
    for evaluator, rows in grouped.items():
        scores = [score for _, score, _ in rows if score is not None]
        reason_counts = Counter(reason for _, _, reasons in rows for reason in reasons)
        total = len(rows)
        passed = sum(1 for ok, _, _ in rows if ok)
        failed = total - passed
        summaries.append(
            EvaluatorSummary(
                evaluator=evaluator,
                total=total,
                passed=passed,
                failed=failed,
                pass_rate=(passed / total) if total else 0.0,
                avg_score=(sum(scores) / len(scores)) if scores else None,
                p50_score=percentile(scores, 0.50),
                p95_score=percentile(scores, 0.95),
                reason_code_counts=dict(reason_counts),
            )
        )
    summaries.sort(key=lambda item: item.evaluator)
    return summaries


def build_dataset_summary(
    *,
    suite_name: str,
    total_records: int,
    sampled_records: int,
    sample_seed: int,
    overall_quality_score: float | None,
    records: list[RecordAssessment],
    dataset_metrics: dict[str, float | int | str | None],
    warnings: list[str] | None = None,
) -> DatasetAssessmentSummary:
    return DatasetAssessmentSummary(
        suite=suite_name,
        total_records=total_records,
        sampled_records=sampled_records,
        sample_seed=sample_seed,
        overall_quality_score=overall_quality_score,
        evaluator_summaries=summarize_evaluators(records),
        dataset_metrics=dataset_metrics,
        warnings=warnings or [],
    )
