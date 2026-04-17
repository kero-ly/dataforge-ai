# src/dataforge/hooks.py
"""Pipeline lifecycle hooks for cross-cutting concerns.

Implement ``PipelineHook`` to receive events during pipeline execution.
Hooks are called in the order they are registered and must not raise exceptions.

Example::

    class CostTracker(PipelineHook):
        async def on_record_completed(self, record: DataRecord) -> None:
            tokens = record.metadata.get("tokens", 0)
            self.total_cost += tokens * PRICE_PER_TOKEN
"""
from __future__ import annotations

from dataforge.schema import DataRecord


class PipelineHook:
    """Base class for pipeline lifecycle hooks.

    Subclass and override any method you care about.
    All methods have no-op defaults so you only implement what you need.
    """

    async def on_pipeline_start(self, *, input_path: str, output_path: str, concurrency: int) -> None:
        """Called once when the pipeline begins execution."""

    async def on_record_started(self, record: DataRecord) -> None:
        """Called immediately before a worker starts processing a record."""

    async def on_record_completed(self, record: DataRecord) -> None:
        """Called after a record passes all evaluators and is written to output."""

    async def on_record_rejected(self, record: DataRecord) -> None:
        """Called when a record is rejected by an evaluator."""

    async def on_record_failed(self, record: DataRecord, error: Exception) -> None:
        """Called when a record fails after exhausting all retries."""

    async def on_pipeline_end(self) -> None:
        """Called once when the pipeline finishes (even if interrupted)."""
