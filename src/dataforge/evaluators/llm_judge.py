# src/dataforge/evaluators/llm_judge.py
from __future__ import annotations

import re
import time
from typing import Any

from dataforge.assessment.schema import AssessmentResult
from dataforge.clients.base import LLMProtocol
from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord


@register_evaluator("llm-judge")
class LLMJudge(BaseEvaluator):
    """Quality evaluator using an LLM as judge (1–5 scoring scale).

    Args:
        llm: Any object with ``async def generate(prompt: str) -> str``.
        criteria: Scoring dimension. Built-in keys: ``"helpfulness"``,
            ``"factuality"``, ``"logical_reasoning"``. Or a custom prompt prefix.
        threshold: Minimum score to accept (inclusive). Default 4.0.
    """

    _BUILTIN_CRITERIA: dict[str, str] = {
        "helpfulness": (
            "Rate the helpfulness and usefulness of the following content "
            "on a scale of 1–5. Reply with only a number."
        ),
        "factuality": (
            "Rate the factual accuracy of the following content "
            "on a scale of 1–5. Reply with only a number."
        ),
        "logical_reasoning": (
            "Rate the logical coherence and reasoning quality of the following content "
            "on a scale of 1–5. Reply with only a number."
        ),
    }

    def __init__(
        self,
        llm: LLMProtocol,
        criteria: str = "helpfulness",
        threshold: float = 4.0,
        eval_max_tokens: int | None = None,
    ) -> None:
        self.llm = llm
        self.threshold = threshold
        self._criteria_prompt = self._BUILTIN_CRITERIA.get(criteria, criteria)
        self.eval_max_tokens = eval_max_tokens

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        started = time.monotonic()
        content = self._get_content(record)
        prompt = f"{self._criteria_prompt}\n\nContent:\n{content}"
        gen_kwargs: dict[str, Any] = {}
        if self.eval_max_tokens is not None:
            gen_kwargs["max_tokens"] = self.eval_max_tokens
        response = await self.llm.generate(prompt, **gen_kwargs)
        score = self._parse_score(response)
        if score is None:
            return AssessmentResult(
                evaluator=type(self).__name__,
                passed=False,
                threshold=self.threshold,
                reason_codes=["score_parse_failed"],
                details={"raw_response": response},
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )
        record.score = score
        criteria_name = next(
            (name for name, prompt_text in self._BUILTIN_CRITERIA.items() if prompt_text == self._criteria_prompt),
            "custom",
        )
        passed = score >= self.threshold
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=score,
            threshold=self.threshold,
            dimension_scores={criteria_name: score},
            reason_codes=[] if passed else ["below_threshold"],
            details={"raw_response": response},
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    @staticmethod
    def _parse_score(text: str) -> float | None:
        stripped = text.strip()
        # Ideal: entire response is just a number (LLM follows instructions)
        exact = re.fullmatch(r"(\d+(?:\.\d+)?)", stripped)
        if exact:
            score = float(exact.group(1))
            return score if 1.0 <= score <= 5.0 else None
        # Fallback: LLM added explanation — collect all numbers, keep those in
        # 1-5 range, return the last one. This correctly ignores large numbers
        # e.g. "Year 2024, score 3" → 2024 out of range, 3 in range → 3.0
        candidates = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", stripped)]
        valid = [s for s in candidates if 1.0 <= s <= 5.0]
        return valid[-1] if valid else None

    def _warmup_prompt(self) -> str:
        """Return a minimal warmup string that shares the judge's KV prefix.

        Used by the pipeline to seed the LLM's prefix KV cache before timing
        starts, so that the first real eval batch hits a hot prefix rather than
        paying the cold-miss penalty.
        """
        return f"{self._criteria_prompt}\n\nContent:\nwarmup"
