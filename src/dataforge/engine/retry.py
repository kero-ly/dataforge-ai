# src/dataforge/engine/retry.py
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class MaxRetriesExceededError(Exception):
    """Raised when an operation fails after exhausting all retry attempts."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        super().__init__(f"Failed after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class RetryEngine:
    """Exponential backoff retry with full jitter.

    Wait formula: min(max_delay, 2^attempt * base_delay) + uniform(0, jitter_cap)

    Args:
        max_retries: Maximum number of retries after the initial attempt.
            Total attempts = max_retries + 1.
        base_delay: Base delay in seconds for exponential backoff.
        max_delay: Maximum delay cap in seconds.
        jitter_cap: Upper bound for random jitter added to each delay.
        retryable_exceptions: Only retry on these exception types.
            Defaults to (Exception,) — retry on any exception.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_cap: float = 1.0,
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_cap = jitter_cap
        self.retryable_exceptions = retryable_exceptions
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, got {base_delay}")
        if max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {max_delay}")
        if jitter_cap < 0:
            raise ValueError(f"jitter_cap must be >= 0, got {jitter_cap}")

    async def run(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute func with exponential backoff retry.

        Args:
            func: An async callable with no arguments to execute.

        Returns:
            The return value of func on success.

        Raises:
            MaxRetriesExceededError: If all attempts fail.
            Any non-retryable exception immediately (not wrapped).
        """
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func()
            except self.retryable_exceptions as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(self.max_delay, (2**attempt) * self.base_delay)
                    delay += random.uniform(0, self.jitter_cap)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, e, delay,
                    )
                    await asyncio.sleep(delay)

        if last_error is None:
            raise RuntimeError("RetryEngine: no error recorded after all attempts failed")
        raise MaxRetriesExceededError(self.max_retries + 1, last_error)
