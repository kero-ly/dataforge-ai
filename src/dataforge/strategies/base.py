# src/dataforge/strategies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

from dataforge.schema import DataRecord


class BaseStrategy(ABC):
    """Transform seed data into synthetic data.

    Subclass this to implement a custom synthesis strategy.

    Example::

        class MyStrategy(BaseStrategy):
            async def apply(self, record: DataRecord) -> DataRecord:
                record.synthetic_data = {"response": await self.llm.generate(...)}
                return record
    """

    @abstractmethod
    async def apply(self, record: DataRecord) -> DataRecord:
        """Apply the synthesis strategy to a single record.

        Args:
            record: The input DataRecord with seed_data populated.

        Returns:
            The same DataRecord with synthetic_data populated.
        """
