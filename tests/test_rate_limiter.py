# tests/test_rate_limiter.py
import time

import pytest

from dataforge.engine.rate_limiter import TokenBucketRateLimiter


async def test_acquire_within_limit():
    limiter = TokenBucketRateLimiter(rpm=60, tpm=10_000)
    # Should not block for the first few requests (bucket starts full)
    start = time.monotonic()
    await limiter.acquire(tokens=10)
    await limiter.acquire(tokens=10)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Expected fast acquire, took {elapsed:.2f}s"


async def test_rpm_bucket_is_full_on_creation():
    limiter = TokenBucketRateLimiter(rpm=10, tpm=10_000)
    # Should be able to make 10 requests immediately without blocking
    start = time.monotonic()
    for _ in range(10):
        await limiter.acquire(tokens=1)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"10 requests should not block, took {elapsed:.2f}s"


async def test_acquire_tokens_exceeding_tpm_raises():
    limiter = TokenBucketRateLimiter(rpm=60, tpm=100)
    with pytest.raises(ValueError, match="exceed"):
        await limiter.acquire(tokens=101)


def test_sync_from_headers_bumps_up():
    limiter = TokenBucketRateLimiter(rpm=100, tpm=10_000)
    # Drain buckets to low values
    limiter._rpm_tokens = 5.0
    limiter._tpm_tokens = 500.0
    limiter.sync_from_headers(remaining_requests=80, remaining_tokens=8000)
    assert limiter._rpm_tokens == 72.0  # 80 * 0.9
    assert limiter._tpm_tokens == 7200.0  # 8000 * 0.9


def test_sync_from_headers_capped_at_max():
    limiter = TokenBucketRateLimiter(rpm=100, tpm=10_000)
    # Drain buckets so sync will try to bump
    limiter._rpm_tokens = 5.0
    limiter._tpm_tokens = 500.0
    # Report remaining > max
    limiter.sync_from_headers(remaining_requests=200, remaining_tokens=20_000)
    assert limiter._rpm_tokens == 100.0  # capped at max rpm
    assert limiter._tpm_tokens == 10_000.0  # capped at max tpm


def test_sync_from_headers_no_downgrade():
    limiter = TokenBucketRateLimiter(rpm=100, tpm=10_000)
    # Buckets are already higher than what server reports
    limiter._rpm_tokens = 90.0
    limiter._tpm_tokens = 9000.0
    limiter.sync_from_headers(remaining_requests=50, remaining_tokens=5000)
    # Should not decrease: 50*0.9=45 < 90, 5000*0.9=4500 < 9000
    assert limiter._rpm_tokens == 90.0
    assert limiter._tpm_tokens == 9000.0


def test_sync_from_headers_none_values():
    limiter = TokenBucketRateLimiter(rpm=100, tpm=10_000)
    limiter._rpm_tokens = 50.0
    limiter._tpm_tokens = 5000.0
    limiter.sync_from_headers(remaining_requests=None, remaining_tokens=None)
    assert limiter._rpm_tokens == 50.0
    assert limiter._tpm_tokens == 5000.0
