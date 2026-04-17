# src/dataforge/evaluators/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import time

from dataforge.assessment.schema import AssessmentResult
from dataforge.schema import DataRecord


class BaseEvaluator(ABC):
    """Quality gate: accepts or rejects a DataRecord.

    Subclass this to implement a custom evaluation strategy.

    Example::

        class MyEvaluator(BaseEvaluator):
            async def evaluate(self, record: DataRecord) -> bool:
                return len(record.synthetic_data.get("response", "")) > 50
    """

    @staticmethod
    def _get_content(record: DataRecord) -> str:
        """Return all synthetic_data values joined as a single string."""
        if record.synthetic_data is None:
            return ""
        return " ".join(str(v) for v in record.synthetic_data.values())

    @abstractmethod
    async def evaluate(self, record: DataRecord) -> bool:
        """Evaluate a record's quality.

        Args:
            record: The DataRecord after synthesis.

        Returns:
            True to accept the record, False to reject it.
        """

    async def assess(self, record: DataRecord) -> AssessmentResult:
        """Return a structured assessment result for the record.

        Subclasses can override this to provide richer dimension scores,
        reason codes, or evaluator-specific details. The default
        implementation wraps ``evaluate()`` for backwards compatibility.
        """
        started = time.monotonic()
        passed = await self.evaluate(record)
        score = record.score
        threshold = getattr(self, "threshold", None)
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=score,
            threshold=float(threshold) if isinstance(threshold, (int, float)) else None,
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )
