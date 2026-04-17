# src/dataforge/evaluators/regex_filter.py
from __future__ import annotations

import json
import re
import time

from dataforge.assessment.schema import AssessmentResult
from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord


@register_evaluator("regex-filter")
class RegexFilter(BaseEvaluator):
    """Fast rule-based pre-filter with optional JSON validation.

    Args:
        blacklist_patterns: Regex patterns; any match → reject.
        require_json: If True, synthetic_data text must contain a valid JSON object.
    """

    def __init__(
        self,
        blacklist_patterns: list[str] | None = None,
        require_json: bool = False,
    ) -> None:
        self._patterns = [re.compile(p) for p in (blacklist_patterns or [])]
        self.require_json = require_json

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        started = time.monotonic()
        text = self._get_content(record)
        matched_patterns: list[str] = []
        reason_codes: list[str] = []
        for pattern in self._patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern)
        if matched_patterns:
            reason_codes.append("regex_blacklist_hit")
        if self.require_json and not self._has_valid_json(text):
            reason_codes.append("json_invalid")
        passed = not reason_codes
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            reason_codes=reason_codes,
            details={"matched_patterns": matched_patterns, "require_json": self.require_json},
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )

    @staticmethod
    def _has_valid_json(text: str) -> bool:
        try:
            json.loads(text.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass
        match = re.search(r"\{[^{}]*\}|\[[^\[\]]*\]", text, re.DOTALL)
        if match:
            try:
                json.loads(match.group())
                return True
            except (json.JSONDecodeError, ValueError):
                pass
        return False
