from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from dataforge.assessment.schema import RecordAssessment
from dataforge.assessment.utils import (
    count_duplicates,
    distinct_n,
    estimate_tokens,
    five_gram_jaccard,
    normalize_text,
)
from dataforge.evaluators.base import BaseEvaluator
from dataforge.evaluators.completeness import CompletenessEvaluator
from dataforge.evaluators.length_window import LengthWindowEvaluator
from dataforge.evaluators.multi_criteria import MultiCriteriaEvaluator
from dataforge.evaluators.regex_filter import RegexFilter
from dataforge.evaluators.similarity import SimilarityEvaluator
from dataforge.registry import register_assessment_suite


class AssessmentSuiteSpec(BaseModel):
    name: str
    description: str
    required_evaluators: list[str]
    optional_evaluators: list[str] = Field(default_factory=list)
    score_weights: dict[str, float]
    dataset_metric_weights: dict[str, float] = Field(default_factory=dict)
    sample_size_default: int = 1000
    sample_seed_default: int = 42
    persist_record_results: bool = True


class AssessmentSuite(Protocol):
    spec: AssessmentSuiteSpec

    def build_evaluators(self, config: Any) -> list[BaseEvaluator]: ...

    async def compute_dataset_metrics(
        self,
        assessed_records: list[RecordAssessment],
        config: Any,
    ) -> dict[str, Any]: ...

    def aggregate_score(
        self,
        assessed_records: list[RecordAssessment],
        dataset_metrics: dict[str, Any],
    ) -> float | None: ...


@dataclass
class SFTReadinessSuite:
    spec: AssessmentSuiteSpec = field(
        default_factory=lambda: AssessmentSuiteSpec(
            name="sft_readiness_v1",
            description="General SFT readiness assessment for instruction-response datasets.",
            required_evaluators=[
                "completeness",
                "length_window",
                "regex_refusal",
                "quality_judge",
                "seed_similarity",
            ],
            score_weights={
                "CompletenessEvaluator": 0.10,
                "LengthWindowEvaluator": 0.10,
                "RegexFilter": 0.10,
                "MultiCriteriaEvaluator": 0.35,
                "SimilarityEvaluator": 0.10,
            },
            dataset_metric_weights={
                "duplicate_rate": 0.10,
                "distinct_2": 0.05,
                "exact_overlap_rate": 0.05,
                "fuzzy_overlap_rate": 0.05,
            },
            sample_size_default=1000,
            sample_seed_default=42,
            persist_record_results=True,
        )
    )

    def build_evaluators(self, config: Any) -> list[BaseEvaluator]:
        judge_llm = getattr(config, "_judge_client", None)
        embedding_cfg = getattr(config, "embedding", None)
        similarity_kwargs: dict[str, Any] = {}
        if embedding_cfg is not None:
            similarity_kwargs = {
                "api_key": embedding_cfg.api_key,
                "base_url": embedding_cfg.base_url,
                "embedding_model": embedding_cfg.model,
            }
        evaluators: list[BaseEvaluator] = [
            CompletenessEvaluator(),
            LengthWindowEvaluator(),
            RegexFilter(
                blacklist_patterns=[
                    r"(?i)as an ai",
                    r"(?i)i cannot",
                    r"(?i)i'm sorry",
                    r"(?i)cannot assist with",
                ]
            ),
        ]
        if judge_llm is not None:
            evaluators.append(
                MultiCriteriaEvaluator(
                    llm=judge_llm,
                    criteria={
                        "helpfulness": 0.5,
                        "accuracy": 0.3,
                        "safety": 0.2,
                    },
                    threshold=3.5,
                )
            )
        if embedding_cfg is not None:
            evaluators.append(
                SimilarityEvaluator(
                    min_similarity=0.2,
                    max_similarity=0.92,
                    **similarity_kwargs,
                )
            )
        return evaluators

    async def compute_dataset_metrics(
        self,
        assessed_records: list[RecordAssessment],
        config: Any,
    ) -> dict[str, Any]:
        responses = [
            str((record.normalized_record.synthetic_data or {}).get("response", ""))
            for record in assessed_records
        ]
        instructions = [
            str(record.normalized_record.seed_data.get("instruction", ""))
            for record in assessed_records
        ]
        duplicate_count, _ = count_duplicates(responses)
        total = len(responses) or 1
        metrics: dict[str, Any] = {
            "duplicate_rate": duplicate_count / total,
            "distinct_1": distinct_n(responses, 1),
            "distinct_2": distinct_n(responses, 2),
            "avg_instruction_tokens": (
                sum(estimate_tokens(text) for text in instructions) / len(instructions)
                if instructions
                else 0.0
            ),
            "avg_response_tokens": (
                sum(estimate_tokens(text) for text in responses) / len(responses)
                if responses
                else 0.0
            ),
        }

        reference_cfg = getattr(config, "reference_corpus", None)
        if reference_cfg is None or not getattr(reference_cfg, "enabled", False):
            return metrics

        exact_hits = 0
        fuzzy_hits = 0
        reference_responses = getattr(config, "_reference_responses", [])
        fuzzy_threshold = float(getattr(reference_cfg, "fuzzy_overlap_threshold", 0.8))
        normalized_reference = {normalize_text(text) for text in reference_responses if normalize_text(text)}
        for response in responses:
            normalized = normalize_text(response)
            if normalized and normalized in normalized_reference:
                exact_hits += 1
                fuzzy_hits += 1
                continue
            if any(five_gram_jaccard(response, ref) >= fuzzy_threshold for ref in reference_responses):
                fuzzy_hits += 1
        metrics["exact_overlap_rate"] = exact_hits / total
        metrics["fuzzy_overlap_rate"] = fuzzy_hits / total
        return metrics

    def aggregate_score(
        self,
        assessed_records: list[RecordAssessment],
        dataset_metrics: dict[str, Any],
    ) -> float | None:
        component_scores: dict[str, float] = {}
        for evaluator_name, weight in self.spec.score_weights.items():
            scores = [
                result.score
                for record in assessed_records
                for result in record.results
                if result.evaluator == evaluator_name and result.score is not None
            ]
            if not scores:
                continue
            if evaluator_name == "MultiCriteriaEvaluator":
                component_scores[evaluator_name] = (sum(scores) / len(scores) / 5.0) * 100.0
            else:
                component_scores[evaluator_name] = (sum(scores) / len(scores)) * 100.0
        for metric_name, weight in self.spec.dataset_metric_weights.items():
            if metric_name not in dataset_metrics:
                continue
            value = float(dataset_metrics[metric_name])
            if metric_name in {"duplicate_rate", "exact_overlap_rate", "fuzzy_overlap_rate"}:
                component_scores[metric_name] = (1.0 - value) * 100.0
            else:
                component_scores[metric_name] = value * 100.0

        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in {**self.spec.score_weights, **self.spec.dataset_metric_weights}.items():
            if key not in component_scores:
                continue
            total_weight += weight
            weighted_sum += component_scores[key] * weight
        if total_weight == 0:
            return None
        return round(weighted_sum / total_weight, 2)


register_assessment_suite("sft_readiness_v1")(SFTReadinessSuite())
