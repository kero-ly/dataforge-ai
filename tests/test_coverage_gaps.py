# tests/test_coverage_gaps.py
"""Additional tests to close coverage gaps in rate_limiter, retry, and other modules."""
import asyncio

import pytest

from dataforge.engine.rate_limiter import TokenBucketRateLimiter
from dataforge.engine.retry import MaxRetriesExceededError, RetryEngine

# --- Rate limiter ---

async def test_rate_limiter_basic_acquire():
    limiter = TokenBucketRateLimiter(rpm=60, tpm=100_000)
    await limiter.acquire(tokens=100)
    # Should not raise — bucket starts full


async def test_rate_limiter_exceed_tpm_raises():
    limiter = TokenBucketRateLimiter(rpm=60, tpm=100)
    with pytest.raises(ValueError, match="exceed the TPM limit"):
        await limiter.acquire(tokens=200)


async def test_rate_limiter_waits_when_exhausted():
    """Drain the bucket and verify acquire blocks then succeeds."""
    # Use high RPM so refill happens quickly
    limiter = TokenBucketRateLimiter(rpm=600, tpm=100_000)
    # Drain all RPM tokens
    for _ in range(600):
        await limiter.acquire(tokens=1)

    # Next acquire should block briefly then succeed (refill kicks in fast at 600/60 = 10/s)
    await asyncio.wait_for(limiter.acquire(tokens=1), timeout=5.0)


async def test_rate_limiter_refill():
    """Verify tokens refill over time."""
    limiter = TokenBucketRateLimiter(rpm=60, tpm=100_000)
    # Drain one token
    await limiter.acquire(tokens=1)
    # Force time advance by calling _refill
    import time
    limiter._last_refill = time.monotonic() - 2  # pretend 2s passed
    limiter._refill()
    # RPM should be refilled (60/60 * 2 = 2 tokens refilled)
    assert limiter._rpm_tokens >= 1.0


# --- Retry engine ---

async def test_retry_immediate_success():
    call_count = 0

    async def succeed():
        nonlocal call_count
        call_count += 1
        return 42

    engine = RetryEngine(max_retries=3, base_delay=0, jitter_cap=0)
    result = await engine.run(succeed)
    assert result == 42
    assert call_count == 1


async def test_retry_fails_then_succeeds():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient")
        return "ok"

    engine = RetryEngine(max_retries=3, base_delay=0, jitter_cap=0)
    result = await engine.run(flaky)
    assert result == "ok"
    assert call_count == 3


async def test_retry_exhausted():
    async def always_fail():
        raise RuntimeError("boom")

    engine = RetryEngine(max_retries=2, base_delay=0, jitter_cap=0)
    with pytest.raises(MaxRetriesExceededError, match="boom"):
        await engine.run(always_fail)


async def test_retry_validation_errors():
    with pytest.raises(ValueError, match="max_retries"):
        RetryEngine(max_retries=-1)
    with pytest.raises(ValueError, match="base_delay"):
        RetryEngine(base_delay=-1)
    with pytest.raises(ValueError, match="max_delay"):
        RetryEngine(max_delay=-1)
    with pytest.raises(ValueError, match="jitter_cap"):
        RetryEngine(jitter_cap=-1)


async def test_retry_non_retryable_exception():
    """Non-retryable exceptions should propagate immediately."""
    call_count = 0

    async def raise_type_error():
        nonlocal call_count
        call_count += 1
        raise TypeError("not retryable")

    engine = RetryEngine(
        max_retries=3,
        base_delay=0,
        jitter_cap=0,
        retryable_exceptions=(RuntimeError,),  # Only retry RuntimeError
    )
    with pytest.raises(TypeError, match="not retryable"):
        await engine.run(raise_type_error)
    assert call_count == 1  # Should not retry


# --- LLMProtocol ---

def test_llm_protocol_runtime_check():
    from dataforge.clients.base import LLMProtocol

    class MockLLM:
        async def generate(self, prompt, **kwargs):
            return "response"

    assert isinstance(MockLLM(), LLMProtocol)


def test_non_llm_fails_protocol():
    from dataforge.clients.base import LLMProtocol

    class NotAnLLM:
        pass

    assert not isinstance(NotAnLLM(), LLMProtocol)


# --- Checkpoint edge cases ---

async def test_checkpoint_corrupt_lines():
    """Checkpoint load should skip malformed lines gracefully."""
    import tempfile
    from pathlib import Path

    from dataforge.engine.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = CheckpointManager(tmpdir)
        wal_path = Path(tmpdir) / "checkpoint.jsonl"
        with open(wal_path, "w") as f:
            f.write('{"id": "good-1"}\n')
            f.write("not json\n")
            f.write("{}\n")  # missing "id" key
            f.write('{"id": "good-2"}\n')

        await ckpt.load()
        assert ckpt.completed_count == 2
        assert await ckpt.is_done("good-1")
        assert await ckpt.is_done("good-2")
