# src/dataforge/clients/openai_client.py
from __future__ import annotations

import logging
import time

import openai

from dataforge.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """LLM client for OpenAI-compatible APIs (OpenAI, Azure OpenAI, etc.).

    Uses the official openai Python SDK with async support.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._aclient = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )

    @staticmethod
    def _estimate_tokens(prompt: str | list[dict]) -> int:
        """Estimate token count from prompt.

        Uses ~5 chars per token to avoid over-counting (which causes
        premature TPM throttling).  The OpenAI tokeniser averages ~4 chars
        per token for English, but prompts contain template boilerplate and
        whitespace that inflate character counts disproportionately.
        """
        if isinstance(prompt, str):
            return max(1, len(prompt) // 5)
        total_chars = sum(len(str(m.get("content", ""))) for m in prompt)
        return max(1, total_chars // 5)

    async def generate(self, prompt: str | list[dict], **kwargs) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Either a string (wrapped in a user message) or a list of
                message dicts (passed directly to the API).
            **kwargs: Additional generation parameters that override generation_kwargs.

        Returns:
            The generated text content.
        """
        has_observers = bool(self._request_observers)
        if self._disable_rate_limit:
            estimated_tokens = 0
        else:
            estimated_tokens = self._estimate_tokens(prompt)
            await self._acquire_rate_limit(tokens=estimated_tokens)
        if has_observers:
            if not estimated_tokens:
                estimated_tokens = self._estimate_tokens(prompt)
            self._notify_request_start(
                prompt=prompt,
                estimated_prompt_tokens=estimated_tokens,
            )
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        gen_kwargs = {**self.generation_kwargs, **kwargs}
        logger.debug("Calling %s model=%s", self.base_url or "openai", self.model)
        started_at = time.monotonic()
        try:
            response = await self._aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                **gen_kwargs,
            )
            if not self._disable_rate_limit:
                self._sync_rate_limit_headers(response)

            content = response.choices[0].message.content
            if content is None:
                raise ValueError(
                    "OpenAI response contained no text content "
                    f"(finish_reason={response.choices[0].finish_reason!r})"
                )

            if has_observers:
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                self._notify_request_end(
                    prompt=prompt,
                    output=content,
                    estimated_prompt_tokens=estimated_tokens,
                    latency_seconds=time.monotonic() - started_at,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            return content
        except Exception as exc:
            if has_observers:
                self._notify_request_error(
                    prompt=prompt,
                    estimated_prompt_tokens=estimated_tokens,
                    latency_seconds=time.monotonic() - started_at,
                    error=exc,
                )
            raise

    async def generate_raw(self, messages: list[dict], **kwargs) -> str:
        """Zero-overhead generate: no observers, no rate limiting, no logging.

        Designed for burst-mode hot paths where per-request overhead matters.
        Callers must provide pre-built message dicts.
        """
        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _sync_rate_limit_headers(self, response) -> None:
        """Extract rate limit headers from API response and sync to rate limiter."""
        try:
            # The OpenAI SDK stores the raw httpx response
            raw = getattr(response, '_response', None)
            if raw is None:
                return
            headers = getattr(raw, 'headers', None)
            if headers is None:
                return
            remaining_requests = headers.get('x-ratelimit-remaining-requests')
            remaining_tokens = headers.get('x-ratelimit-remaining-tokens')
            if remaining_requests is not None or remaining_tokens is not None:
                self._rate_limiter.sync_from_headers(
                    remaining_requests=int(remaining_requests) if remaining_requests else None,
                    remaining_tokens=int(remaining_tokens) if remaining_tokens else None,
                )
        except (ValueError, TypeError, AttributeError):
            pass  # Silently ignore if headers not available
