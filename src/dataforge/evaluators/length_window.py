from __future__ import annotations

from dataforge.assessment.schema import AssessmentResult
from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord


def _estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


@register_evaluator("length-window")
class LengthWindowEvaluator(BaseEvaluator):
    """Check instruction/response token windows for SFT readiness."""

    def __init__(
        self,
        min_instruction_tokens: int = 3,
        max_instruction_tokens: int = 512,
        min_response_tokens: int = 10,
        max_response_tokens: int = 2048,
    ) -> None:
        self.min_instruction_tokens = min_instruction_tokens
        self.max_instruction_tokens = max_instruction_tokens
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        instruction = str(record.seed_data.get("instruction", ""))
        response = str((record.synthetic_data or {}).get("response", ""))
        instruction_tokens = _estimate_tokens(instruction)
        response_tokens = _estimate_tokens(response)
        reason_codes: list[str] = []
        if instruction_tokens < self.min_instruction_tokens:
            reason_codes.append("instruction_too_short")
        if instruction_tokens > self.max_instruction_tokens:
            reason_codes.append("instruction_too_long")
        if response_tokens < self.min_response_tokens:
            reason_codes.append("response_too_short")
        if response_tokens > self.max_response_tokens:
            reason_codes.append("response_too_long")
        passed = not reason_codes
        score = 1.0 if passed else 0.0
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=score,
            threshold=1.0,
            reason_codes=reason_codes,
            details={
                "instruction_tokens": instruction_tokens,
                "response_tokens": response_tokens,
                "instruction_window": [
                    self.min_instruction_tokens,
                    self.max_instruction_tokens,
                ],
                "response_window": [
                    self.min_response_tokens,
                    self.max_response_tokens,
                ],
            },
        )
