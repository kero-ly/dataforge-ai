# src/dataforge/evaluators/multi_criteria.py
"""Multi-criteria evaluator: evaluates multiple dimensions in a single LLM call."""
from __future__ import annotations

import json
import logging
import re
import time

from dataforge.assessment.schema import AssessmentResult
from dataforge.clients.base import LLMProtocol
from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Evaluate the following content on these dimensions. For each dimension, \
provide a score from 1 to 5 (1 = very poor, 5 = excellent).

Dimensions:
{dimensions}

Content:
{content}

Reply with ONLY a JSON object mapping each dimension name to its integer score. Example:
{example}
"""


@register_evaluator("multi-criteria")
class MultiCriteriaEvaluator(BaseEvaluator):
    """Evaluates content on multiple dimensions in a single LLM call.

    Instead of chaining multiple LLMJudge evaluators (which costs N API calls),
    this evaluator asks the LLM to rate multiple dimensions at once and computes
    a weighted average score.

    Args:
        llm: Any object with ``async def generate(prompt: str) -> str``.
        criteria: Mapping from dimension name to weight. Weights are normalised
            internally so they don't need to sum to 1.
        threshold: Minimum weighted-average score to accept (inclusive).
        max_repair_attempts: Number of JSON re-parse attempts on malformed output.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        criteria: dict[str, float] | None = None,
        threshold: float = 3.5,
        max_repair_attempts: int = 1,
    ) -> None:
        self.llm = llm
        if criteria is not None and not criteria:
            raise ValueError("criteria must contain at least one dimension")
        self.criteria: dict[str, float] = criteria if criteria is not None else {
            "helpfulness": 1.0,
            "accuracy": 1.0,
            "safety": 1.0,
        }
        self.threshold = threshold
        self.max_repair_attempts = max_repair_attempts

        # Pre-compute normalised weights
        total_weight = sum(self.criteria.values())
        self._weights = {k: v / total_weight for k, v in self.criteria.items()}

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        started = time.monotonic()
        content = self._get_content(record)
        dimensions = "\n".join(f"- {name}" for name in self.criteria)
        example = json.dumps({name: 4 for name in self.criteria})
        prompt = _PROMPT_TEMPLATE.format(
            dimensions=dimensions,
            content=content,
            example=example,
        )

        response = await self.llm.generate(prompt)
        scores = self._parse_scores(response, attempt=0)

        if scores is None:
            logger.warning("Failed to parse multi-criteria scores for record %s", record.id)
            return AssessmentResult(
                evaluator=type(self).__name__,
                passed=False,
                threshold=self.threshold,
                reason_codes=["score_parse_failed"],
                details={"raw_response": response},
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )

        # Store per-dimension scores in metadata
        record.metadata["criteria_scores"] = scores

        # Compute weighted average
        weighted_sum = sum(
            scores.get(dim, 0.0) * weight
            for dim, weight in self._weights.items()
        )
        record.score = round(weighted_sum, 4)
        record.metadata["weighted_score"] = record.score

        passed = record.score >= self.threshold
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=record.score,
            threshold=self.threshold,
            dimension_scores={key: float(value) for key, value in scores.items()},
            reason_codes=[] if passed else ["below_threshold"],
            details={"raw_response": response, "weights": dict(self._weights)},
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    def _parse_scores(
        self, text: str, attempt: int
    ) -> dict[str, float] | None:
        """Parse JSON scores from LLM response."""
        # Strip markdown fences
        cleaned = text.strip()
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Try to find a JSON object
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            if attempt < self.max_repair_attempts:
                logger.debug("JSON parse failed (attempt %d), trying fallback extraction", attempt)
                return self._fallback_parse(text)
            return None

        if not isinstance(data, dict):
            return None

        # Validate: every value must be a number in 1-5
        scores: dict[str, float] = {}
        for dim in self.criteria:
            val = data.get(dim)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if 1.0 <= fval <= 5.0:
                scores[dim] = fval

        return scores if scores else None

    def _fallback_parse(self, text: str) -> dict[str, float] | None:
        """Extract scores from unstructured text by searching for dimension names."""
        scores: dict[str, float] = {}
        for dim in self.criteria:
            # Look for patterns like "helpfulness: 4" or "helpfulness = 4"
            pattern = rf"{re.escape(dim)}\s*[:=]\s*(\d+(?:\.\d+)?)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 1.0 <= val <= 5.0:
                    scores[dim] = val
        return scores if scores else None
