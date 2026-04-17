# tests/test_concurrency.py
from __future__ import annotations

import pytest

from dataforge.engine.concurrency import AdaptiveSemaphore


@pytest.mark.asyncio
async def test_adaptive_semaphore_basic_acquire_release():
    """Basic async context manager works."""
    sem = AdaptiveSemaphore(initial=2, min_concurrency=1, max_concurrency=10)
    assert sem.current_concurrency == 2

    async with sem:
        # We should be inside the semaphore
        pass

    # Should be able to acquire again after release
    async with sem:
        pass


@pytest.mark.asyncio
async def test_adaptive_semaphore_calibration_phase():
    """During first N requests, no adjustment happens."""
    window = 10
    sem = AdaptiveSemaphore(
        initial=5,
        min_concurrency=1,
        max_concurrency=50,
        calibration_window=window,
    )
    assert sem.latency_target is None
    assert sem.current_concurrency == 5

    # Report fewer latencies than calibration window — no target set
    for _ in range(window - 1):
        sem.report_latency(0.1)

    assert sem.latency_target is None
    assert sem.current_concurrency == 5  # No change during calibration

    # One more to complete calibration
    sem.report_latency(0.1)
    assert sem.latency_target is not None
    assert sem.current_concurrency == 5  # Still no change right after calibration


@pytest.mark.asyncio
async def test_adaptive_semaphore_increase_on_low_latency():
    """After calibration, low latency increases concurrency."""
    sem = AdaptiveSemaphore(
        initial=5,
        min_concurrency=1,
        max_concurrency=50,
        latency_target=1.0,  # explicit target, skip calibration
        increase_rate=2.0,
    )
    assert sem.current_concurrency == 5

    # Report low latency (below target * 1.5 = 1.5).
    # Adjustment happens every 5th report.
    for _ in range(5):
        sem.report_latency(0.5)

    assert sem.current_concurrency == 7  # 5 + 2.0 = 7


@pytest.mark.asyncio
async def test_adaptive_semaphore_decrease_on_high_latency():
    """After calibration, high latency decreases concurrency."""
    sem = AdaptiveSemaphore(
        initial=10,
        min_concurrency=1,
        max_concurrency=50,
        latency_target=1.0,  # explicit target
        decrease_factor=0.5,
    )
    assert sem.current_concurrency == 10

    # Report high latency (above target * 2.5 = 2.5).
    # Adjustment happens every 5th report.
    for _ in range(5):
        sem.report_latency(3.0)

    assert sem.current_concurrency == 5  # 10 * 0.5 = 5


@pytest.mark.asyncio
async def test_adaptive_semaphore_respects_min_max():
    """Never goes below min or above max."""
    # Test max
    sem_max = AdaptiveSemaphore(
        initial=9,
        min_concurrency=1,
        max_concurrency=10,
        latency_target=1.0,
        increase_rate=5.0,
    )
    for _ in range(5):
        sem_max.report_latency(0.1)  # low latency, wants to increase
    assert sem_max.current_concurrency <= 10

    # Test min
    sem_min = AdaptiveSemaphore(
        initial=3,
        min_concurrency=2,
        max_concurrency=50,
        latency_target=1.0,
        decrease_factor=0.1,
    )
    for _ in range(5):
        sem_min.report_latency(10.0)  # very high latency, wants to decrease
    assert sem_min.current_concurrency >= 2


@pytest.mark.asyncio
async def test_adaptive_semaphore_with_explicit_target():
    """When latency_target provided, skip calibration."""
    sem = AdaptiveSemaphore(
        initial=5,
        min_concurrency=1,
        max_concurrency=50,
        latency_target=0.5,
    )
    # Should already be calibrated
    assert sem.latency_target == 0.5

    # First report should go to adjustment logic, not calibration
    # (No crash, no calibration list growth)
    sem.report_latency(0.1)
    assert sem.latency_target == 0.5  # unchanged
