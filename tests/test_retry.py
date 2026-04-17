# tests/test_retry.py
from unittest.mock import patch

import pytest

from dataforge.engine.retry import MaxRetriesExceededError, RetryEngine


async def test_retry_succeeds_on_second_attempt():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("temporary error")
        return "success"

    engine = RetryEngine(max_retries=3, base_delay=0.01)
    result = await engine.run(flaky)
    assert result == "success"
    assert call_count == 2


async def test_retry_raises_after_max_retries():
    async def always_fails():
        raise ValueError("always fails")

    engine = RetryEngine(max_retries=2, base_delay=0.01)
    with pytest.raises(MaxRetriesExceededError) as exc_info:
        await engine.run(always_fails)
    assert exc_info.value.attempts == 3  # initial + 2 retries
    assert isinstance(exc_info.value.last_error, ValueError)
    assert str(exc_info.value.last_error) == "always fails"


async def test_retry_delay_is_exponential():
    delays = []

    async def mock_sleep(t):
        delays.append(t)

    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ValueError("fail")
        return "ok"

    engine = RetryEngine(max_retries=4, base_delay=1.0, max_delay=60.0, jitter_cap=0.0)
    with patch("asyncio.sleep", side_effect=mock_sleep):
        result = await engine.run(flaky)

    assert result == "ok"
    # With jitter_cap=0.0, delays should be exactly exponential: 1s, 2s, 4s
    assert len(delays) == 3
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert delays[2] == pytest.approx(4.0)


async def test_only_retries_specified_exceptions():
    call_count = 0

    async def fails_with_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    # Only retry on KeyError, not ValueError
    engine = RetryEngine(
        max_retries=3,
        base_delay=0.01,
        retryable_exceptions=(KeyError,),
    )
    with pytest.raises(ValueError, match="not retryable"):
        await engine.run(fails_with_value_error)

    # Should have been called exactly once (no retries)
    assert call_count == 1
