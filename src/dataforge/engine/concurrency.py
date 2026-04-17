# src/dataforge/engine/concurrency.py
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class AdaptiveSemaphore:
    """AIMD-based adaptive concurrency controller.

    Wraps an asyncio.Semaphore that dynamically adjusts its capacity
    based on request latency feedback.

    - When latency is below threshold: slowly increase concurrency (additive)
    - When latency exceeds threshold: quickly reduce concurrency (multiplicative)

    Usage::

        controller = AdaptiveSemaphore(initial=10, min_concurrency=1, max_concurrency=100)
        async with controller:
            result = await do_work()
            controller.report_latency(elapsed)
    """

    def __init__(
        self,
        initial: int = 10,
        min_concurrency: int = 1,
        max_concurrency: int = 200,
        latency_target: float | None = None,  # auto-calibrated if None
        increase_rate: float = 1.0,
        decrease_factor: float = 0.7,
        calibration_window: int = 20,
    ) -> None:
        self._current = float(initial)
        self._min = min_concurrency
        self._max = max_concurrency
        self._latency_target = latency_target
        self._increase_rate = increase_rate
        self._decrease_factor = decrease_factor
        self._calibration_window = calibration_window

        self._sem = asyncio.Semaphore(initial)
        self._calibration_latencies: list[float] = []
        self._calibrated = latency_target is not None
        self._adjust_counter = 0

    @property
    def current_concurrency(self) -> int:
        return int(self._current)

    @property
    def latency_target(self) -> float | None:
        return self._latency_target

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self._sem.release()

    def report_latency(self, latency: float) -> None:
        """Report a completed request's latency for concurrency adjustment."""
        # Calibration phase: collect initial latencies to set target
        if not self._calibrated:
            self._calibration_latencies.append(latency)
            if len(self._calibration_latencies) >= self._calibration_window:
                sorted_lats = sorted(self._calibration_latencies)
                # Use p50 as target
                self._latency_target = sorted_lats[len(sorted_lats) // 2]
                self._calibrated = True
                logger.info(
                    "Adaptive concurrency calibrated: target_latency=%.3fs from %d samples",
                    self._latency_target,
                    len(self._calibration_latencies),
                )
            return

        assert self._latency_target is not None
        self._adjust_counter += 1

        # Only adjust every N requests to avoid oscillation
        if self._adjust_counter % 5 != 0:
            return

        old = self._current
        if latency < self._latency_target * 1.5:
            # Additive increase: slowly grow
            self._current = min(self._max, self._current + self._increase_rate)
        elif latency > self._latency_target * 2.5:
            # Multiplicative decrease: quickly shrink
            self._current = max(self._min, self._current * self._decrease_factor)

        new_int = int(self._current)
        old_int = int(old)
        if new_int != old_int:
            self._resize_semaphore(old_int, new_int)
            logger.info(
                "Adaptive concurrency: %d -> %d (latency=%.3fs, target=%.3fs)",
                old_int,
                new_int,
                latency,
                self._latency_target,
            )

    def _resize_semaphore(self, old_size: int, new_size: int) -> None:
        """Resize the semaphore by releasing or acquiring permits."""
        diff = new_size - old_size
        if diff > 0:
            for _ in range(diff):
                self._sem.release()
        elif diff < 0:
            # For decrease: we reduce _sem._value directly
            # This is safe in single-threaded asyncio
            self._sem._value = max(0, self._sem._value + diff)
