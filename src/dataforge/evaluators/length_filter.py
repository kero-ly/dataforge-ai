# src/dataforge/evaluators/length_filter.py
from __future__ import annotations

from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord


@register_evaluator("length-filter")
class LengthFilter(BaseEvaluator):
    """Filter records by text length boundaries.

    Checks the character count of a specified field in ``synthetic_data``
    (or all fields joined if ``field`` is not set).

    Args:
        min_length: Minimum character count (inclusive). Default 0 (no minimum).
        max_length: Maximum character count (inclusive). Default None (no maximum).
        field: Specific key in ``synthetic_data`` to check. If None, all values
            are joined and checked as a single string.
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int | None = None,
        field: str | None = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.field = field

    async def evaluate(self, record: DataRecord) -> bool:
        if self.field is not None:
            text = str((record.synthetic_data or {}).get(self.field, ""))
        else:
            text = self._get_content(record)

        length = len(text)
        if length < self.min_length:
            return False
        return not (self.max_length is not None and length > self.max_length)
