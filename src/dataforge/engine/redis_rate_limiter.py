# src/dataforge/engine/redis_rate_limiter.py
"""Redis-backed distributed rate limiter for coordinated RPM/TPM limits.

When multiple workers share a single API key, this rate limiter
coordinates their requests through Redis to stay within the global
RPM and TPM budget.

Uses Lua scripts executed atomically to implement sliding-window
rate limiting for RPM and a decaying counter for TPM.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def _import_redis():
    try:
        import redis.asyncio as aioredis
        return aioredis
    except ImportError:
        raise ImportError(
            "redis is required for distributed rate limiting. "
            "Install it with: pip install 'dataforge[distributed]'"
        ) from None


# Lua script for atomic rate-limit check-and-acquire.
# KEYS[1] = RPM sorted set key
# KEYS[2] = TPM counter key
# ARGV[1] = current timestamp (float seconds)
# ARGV[2] = tokens to consume
# ARGV[3] = RPM limit
# ARGV[4] = TPM limit
# Returns: 0 if acquired, or wait_ms (positive integer) if rate-limited.
_ACQUIRE_SCRIPT = """
local rpm_key = KEYS[1]
local tpm_key = KEYS[2]
local now = tonumber(ARGV[1])
local tokens = tonumber(ARGV[2])
local rpm_limit = tonumber(ARGV[3])
local tpm_limit = tonumber(ARGV[4])

-- Clean up RPM entries older than 60 seconds
local window_start = now - 60
redis.call('ZREMRANGEBYSCORE', rpm_key, '-inf', window_start)

-- Check RPM
local rpm_count = redis.call('ZCARD', rpm_key)
if rpm_count >= rpm_limit then
    -- Find the oldest entry to calculate wait time
    local oldest = redis.call('ZRANGE', rpm_key, 0, 0, 'WITHSCORES')
    if #oldest >= 2 then
        local wait = (tonumber(oldest[2]) + 60) - now
        if wait > 0 then
            return math.ceil(wait * 1000)
        end
    end
    return 1000
end

-- Check TPM
local tpm_used = tonumber(redis.call('GET', tpm_key) or '0')
if tpm_used + tokens > tpm_limit then
    -- TPM exceeded, need to wait for the counter to decay
    local ttl = redis.call('TTL', tpm_key)
    if ttl > 0 then
        return ttl * 1000
    end
    return 1000
end

-- Acquire: add RPM entry and increment TPM
redis.call('ZADD', rpm_key, now, now .. ':' .. math.random(1000000))
redis.call('EXPIRE', rpm_key, 120)
redis.call('INCRBY', tpm_key, tokens)
-- Set TTL on TPM key if not already set (60-second window)
if redis.call('TTL', tpm_key) < 0 then
    redis.call('EXPIRE', tpm_key, 60)
end

return 0
"""


class RedisRateLimiter:
    """Distributed rate limiter using Redis for cross-worker coordination.

    Enforces global RPM and TPM limits across all workers sharing
    the same ``limiter_id``.

    Usage::

        limiter = RedisRateLimiter(
            redis_url="redis://localhost:6379",
            rpm=500,
            tpm=100_000,
            limiter_id="openai-gpt4",
        )
        await limiter.connect()
        await limiter.acquire(tokens=150)  # blocks until budget available
        await limiter.close()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        rpm: int = 60,
        tpm: int = 100_000,
        limiter_id: str = "default",
    ) -> None:
        self._redis_url = redis_url
        self._rpm = rpm
        self._tpm = tpm
        self._limiter_id = limiter_id
        self._rpm_key = f"dataforge:ratelimit:{limiter_id}:rpm"
        self._tpm_key = f"dataforge:ratelimit:{limiter_id}:tpm"
        self._redis = None
        self._script = None

    async def connect(self) -> None:
        """Initialize Redis connection and register Lua script."""
        aioredis = _import_redis()
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        self._script = self._redis.register_script(_ACQUIRE_SCRIPT)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire rate limit tokens, blocking until budget is available.

        Args:
            tokens: Number of tokens to consume (for TPM tracking).
        """
        assert self._redis is not None and self._script is not None, (
            "Call connect() first"
        )

        while True:
            now = time.time()
            result = await self._script(
                keys=[self._rpm_key, self._tpm_key],
                args=[now, tokens, self._rpm, self._tpm],
            )
            if result == 0:
                return  # acquired successfully
            # Wait the suggested time before retrying
            wait_seconds = max(int(result) / 1000.0, 0.05)
            logger.debug(
                "Rate limited (limiter=%s), waiting %.1fs",
                self._limiter_id, wait_seconds,
            )
            import asyncio
            await asyncio.sleep(wait_seconds)
