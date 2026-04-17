from __future__ import annotations

from dataforge.assessment.schema import AssessmentResult
from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord


@register_evaluator("completeness")
class CompletenessEvaluator(BaseEvaluator):
    """Check that the normalized record contains non-empty instruction and response."""

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        instruction = str(record.seed_data.get("instruction", "")).strip()
        response = str((record.synthetic_data or {}).get("response", "")).strip()
        reason_codes: list[str] = []
        if not instruction:
            reason_codes.append("missing_instruction")
        if not response:
            reason_codes.append("missing_response")
        passed = not reason_codes
        score = 1.0 if passed else 0.0
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=score,
            threshold=1.0,
            reason_codes=reason_codes,
            details={
                "instruction_present": bool(instruction),
                "response_present": bool(response),
            },
        )
