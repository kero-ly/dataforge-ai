# src/dataforge/clients/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from dataforge.engine.rate_limiter import TokenBucketRateLimiter

# Minimal structural type for any object that can generate text.
# Use this in strategy/evaluator __init__ signatures instead of ``Any``.
ChatMessage = dict[str, str]


@runtime_checkable
class LLMProtocol(Protocol):
    """Structural type for LLM clients used by strategies and evaluators."""

    async def generate(self, prompt: str | list[ChatMessage], **kwargs: Any) -> str: ...


@runtime_checkable
class LLMRequestObserver(Protocol):
    """Observer interface for request-level LLM metrics."""

    def on_request_start(
        self,
        *,
        model: str,
        prompt: str | list[ChatMessage],
        estimated_prompt_tokens: int,
    ) -> None: ...

    def on_request_end(
        self,
        *,
        model: str,
        prompt: str | list[ChatMessage],
        output: str,
        estimated_prompt_tokens: int,
        latency_seconds: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None: ...

    def on_request_error(
        self,
        *,
        model: str,
        prompt: str | list[ChatMessage],
        estimated_prompt_tokens: int,
        latency_seconds: float,
        error: Exception,
    ) -> None: ...


class BaseLLMClient(ABC):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        rpm_limit: int = 60,
        tpm_limit: int = 100_000,
        generation_kwargs: dict[str, Any] | None = None,
        disable_rate_limit: bool = False,
        request_timeout: float | None = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.generation_kwargs: dict[str, Any] = generation_kwargs or {}
        self._disable_rate_limit = disable_rate_limit
        self.request_timeout = request_timeout
        self._rate_limiter = TokenBucketRateLimiter(rpm=rpm_limit, tpm=tpm_limit)
        self._request_observers: list[LLMRequestObserver] = []

    def add_observer(self, observer: LLMRequestObserver) -> None:
        """Register a request observer if it is not already present."""
        if observer not in self._request_observers:
            self._request_observers.append(observer)

    def remove_observer(self, observer: LLMRequestObserver) -> None:
        """Unregister a previously registered request observer."""
        if observer in self._request_observers:
            self._request_observers.remove(observer)

    async def _acquire_rate_limit(self, tokens: int = 1) -> None:
        """Acquire rate-limit budget before making an API call.

        Blocks until both RPM and TPM budgets are available.
        Skipped entirely when ``disable_rate_limit=True``.
        Concrete generate() implementations should call this.
        """
        if self._disable_rate_limit:
            return
        await self._rate_limiter.acquire(tokens=tokens)

    def _notify_request_start(
        self,
        *,
        prompt: str | list[ChatMessage],
        estimated_prompt_tokens: int,
    ) -> None:
        for observer in self._request_observers:
            observer.on_request_start(
                model=self.model,
                prompt=prompt,
                estimated_prompt_tokens=estimated_prompt_tokens,
            )

    def _notify_request_end(
        self,
        *,
        prompt: str | list[ChatMessage],
        output: str,
        estimated_prompt_tokens: int,
        latency_seconds: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        for observer in self._request_observers:
            observer.on_request_end(
                model=self.model,
                prompt=prompt,
                output=output,
                estimated_prompt_tokens=estimated_prompt_tokens,
                latency_seconds=latency_seconds,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

    def _notify_request_error(
        self,
        *,
        prompt: str | list[ChatMessage],
        estimated_prompt_tokens: int,
        latency_seconds: float,
        error: Exception,
    ) -> None:
        for observer in self._request_observers:
            observer.on_request_error(
                model=self.model,
                prompt=prompt,
                estimated_prompt_tokens=estimated_prompt_tokens,
                latency_seconds=latency_seconds,
                error=error,
            )

    @abstractmethod
    async def generate(self, prompt: str | list[ChatMessage], **kwargs: Any) -> str:
        """Send a prompt and return the generated text."""
