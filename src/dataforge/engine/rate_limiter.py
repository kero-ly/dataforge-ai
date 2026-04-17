# src/dataforge/engine/rate_limiter.py
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Dual token bucket enforcing both RPM and TPM limits.

    Both buckets start full. acquire() blocks until both buckets have capacity.
    Buckets refill continuously based on elapsed time.

    Note: Instances must be created and used within a single event loop context.
    Creating an instance in one event loop and using it in another will result
    in undefined behavior due to the internal asyncio.Lock.
    """

    def __init__(self, rpm: int, tpm: int) -> None:
        self._rpm = rpm
        self._tpm = tpm

        # Buckets start full
        self._rpm_tokens: float = float(rpm)
        self._tpm_tokens: float = float(tpm)

        self._last_refill = time.monotonic()
        self._lock: asyncio.Lock | None = None

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        # Refill at per-second rate derived from per-minute limit
        self._rpm_tokens = min(float(self._rpm), self._rpm_tokens + elapsed * self._rpm / 60.0)
        self._tpm_tokens = min(float(self._tpm), self._tpm_tokens + elapsed * self._tpm / 60.0)

    def _time_until_available(self, tokens: int) -> float:
        """Seconds until the bucket has capacity for this request."""
        needed_rpm = max(0.0, 1.0 - self._rpm_tokens)
        needed_tpm = max(0.0, float(tokens) - self._tpm_tokens)
        wait_rpm = needed_rpm * 60.0 / self._rpm if needed_rpm > 0 else 0.0
        wait_tpm = needed_tpm * 60.0 / self._tpm if needed_tpm > 0 else 0.0
        return max(wait_rpm, wait_tpm)

    async def acquire(self, tokens: int = 1) -> None:
        """Block until both RPM (1 request slot) and TPM (tokens) budgets are available.

        Uses a fast path when both buckets clearly have surplus capacity,
        avoiding the lock entirely.
        """
        if tokens > self._tpm:
            raise ValueError(
                f"Requested tokens ({tokens}) exceed the TPM limit ({self._tpm}). "
                "A single acquire cannot request more tokens than the bucket capacity."
            )
        # Fast path: when buckets have ample capacity, skip the lock.
        # Safe in single-threaded asyncio because there are no await points
        # between the check and the decrement.
        ftokens = float(tokens)
        self._refill()
        if self._rpm_tokens >= 2.0 and self._tpm_tokens >= ftokens * 2.0:
            self._rpm_tokens -= 1.0
            self._tpm_tokens -= ftokens
            return

        if self._lock is None:
            self._lock = asyncio.Lock()
        while True:
            async with self._lock:
                self._refill()
                if self._rpm_tokens >= 1.0 and self._tpm_tokens >= ftokens:
                    self._rpm_tokens -= 1.0
                    self._tpm_tokens -= ftokens
                    return
                wait_time = self._time_until_available(tokens)
            logger.debug("Rate limit reached, waiting %.3fs (rpm=%.1f, tpm=%.1f)", wait_time, self._rpm_tokens, self._tpm_tokens)
            await asyncio.sleep(wait_time)

    def sync_from_headers(
        self,
        remaining_requests: int | None = None,
        remaining_tokens: int | None = None,
    ) -> None:
        """Sync bucket state from API response headers.

        If the server reports more remaining capacity than our local bucket,
        bump our bucket up (with a safety margin). Never exceed configured max.
        This allows the limiter to be less conservative when the server has headroom.
        """
        if remaining_requests is not None:
            server_rpm = float(remaining_requests) * 0.9  # 10% safety margin
            if server_rpm > self._rpm_tokens:
                self._rpm_tokens = min(float(self._rpm), server_rpm)
        if remaining_tokens is not None:
            server_tpm = float(remaining_tokens) * 0.9
            if server_tpm > self._tpm_tokens:
                self._tpm_tokens = min(float(self._tpm), server_tpm)
