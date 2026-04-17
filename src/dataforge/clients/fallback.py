# src/dataforge/clients/fallback.py
"""Failover LLM client that automatically switches between primary and fallback."""
from __future__ import annotations

import logging
from typing import Any

from dataforge.clients.base import BaseLLMClient, ChatMessage

logger = logging.getLogger(__name__)


class FallbackClient(BaseLLMClient):
    """LLM client with automatic failover from primary to fallback.

    Wraps two BaseLLMClient instances. When the primary client fails
    ``max_failures`` times consecutively, subsequent requests are routed
    to the fallback client. After ``recovery_after`` successful fallback
    calls the primary is retried again.

    Usage::

        primary = OpenAIClient(model="gpt-4o", api_key=key)
        fallback = OpenAIClient(model="gpt-4o-mini", api_key=key)
        client = FallbackClient(primary=primary, fallback=fallback)
        result = await client.generate("Hello")
    """

    def __init__(
        self,
        primary: BaseLLMClient,
        fallback: BaseLLMClient,
        max_failures: int = 3,
        recovery_after: int = 10,
    ) -> None:
        # We don't call super().__init__ because FallbackClient delegates
        # all rate limiting to the wrapped clients.
        self.model = primary.model
        self._primary = primary
        self._fallback = fallback
        self._max_failures = max_failures
        self._recovery_after = recovery_after
        self._consecutive_failures = 0
        self._fallback_successes = 0
        self._request_observers = []

    @property
    def using_fallback(self) -> bool:
        """Whether the client is currently routing to the fallback."""
        return self._consecutive_failures >= self._max_failures

    def add_observer(self, observer) -> None:
        super().add_observer(observer)
        self._primary.add_observer(observer)
        self._fallback.add_observer(observer)

    async def generate(
        self, prompt: str | list[ChatMessage], **kwargs: Any
    ) -> str:
        """Generate text, automatically failing over if primary is unhealthy."""
        # Try primary if it hasn't exceeded failure threshold
        if not self.using_fallback:
            try:
                result = await self._primary.generate(prompt, **kwargs)
                self._consecutive_failures = 0
                return result
            except Exception as exc:
                self._consecutive_failures += 1
                logger.warning(
                    "Primary client failed (%d/%d): %s",
                    self._consecutive_failures,
                    self._max_failures,
                    str(exc)[:200],
                )
                if self.using_fallback:
                    logger.warning(
                        "Switching to fallback client after %d consecutive failures",
                        self._max_failures,
                    )

        # Use fallback
        result = await self._fallback.generate(prompt, **kwargs)
        self._fallback_successes += 1

        # Periodically retry primary
        if self._fallback_successes >= self._recovery_after:
            logger.info(
                "Attempting recovery: retrying primary after %d fallback successes",
                self._fallback_successes,
            )
            self._consecutive_failures = 0
            self._fallback_successes = 0

        return result
