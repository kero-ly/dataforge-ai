from __future__ import annotations

import hashlib
from typing import Any

from dataforge.clients.base import BaseLLMClient, ChatMessage, LLMRequestObserver
from dataforge.clients.vllm_client import vLLMClient


class vLLMClusterClient(BaseLLMClient):
    """Client over multiple OpenAI-compatible vLLM endpoints."""

    def __init__(
        self,
        model: str,
        base_urls: list[str],
        rpm_limit: int = 5000,
        tpm_limit: int = 10_000_000,
        disable_rate_limit: bool = True,
        generation_kwargs: dict[str, Any] | None = None,
        routing_strategy: str = "round_robin",
        prefix_replication: int = 2,
        prefix_chars: int = 96,
        **kwargs: Any,
    ) -> None:
        normalized_urls = [url.strip() for url in base_urls if url.strip()]
        if not normalized_urls:
            raise ValueError("base_urls must contain at least one non-empty URL")
        if routing_strategy not in {"round_robin", "prefix_affinity"}:
            raise ValueError(
                "routing_strategy must be 'round_robin' or 'prefix_affinity'"
            )

        self.base_urls = normalized_urls
        self.routing_strategy = routing_strategy
        self.prefix_replication = max(1, prefix_replication)
        self.prefix_chars = max(1, prefix_chars)
        self._clients = [
            vLLMClient(
                model=model,
                base_url=base_url,
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                disable_rate_limit=disable_rate_limit,
                generation_kwargs=generation_kwargs,
                **kwargs,
            )
            for base_url in normalized_urls
        ]
        self._next_client_idx = 0
        self._prefix_offsets: dict[str, int] = {}
        super().__init__(
            model=model,
            api_key="not-needed",
            base_url=",".join(normalized_urls),
            rpm_limit=rpm_limit * len(normalized_urls),
            tpm_limit=tpm_limit * len(normalized_urls),
            generation_kwargs=generation_kwargs,
            disable_rate_limit=disable_rate_limit,
        )

    def _pick_client(self) -> vLLMClient:
        client = self._clients[self._next_client_idx]
        self._next_client_idx = (self._next_client_idx + 1) % len(self._clients)
        return client

    def _prompt_prefix_key(self, prompt: str | list[ChatMessage]) -> str:
        if isinstance(prompt, str):
            return prompt[: self.prefix_chars]
        parts: list[str] = []
        total_len = 0
        for message in prompt:
            role = message.get("role", "")
            content = str(message.get("content", ""))
            parts.append(role)
            parts.append(content)
            total_len += len(role) + len(content)
            if total_len >= self.prefix_chars:
                break
        return "".join(parts)[: self.prefix_chars]

    def _pick_client_for_prompt(self, prompt: str | list[ChatMessage]) -> vLLMClient:
        if self.routing_strategy == "round_robin":
            return self._pick_client()

        prefix = self._prompt_prefix_key(prompt)
        if not prefix:
            return self._pick_client()

        digest = hashlib.blake2b(prefix.encode("utf-8"), digest_size=8).digest()
        start = int.from_bytes(digest, "big") % len(self._clients)
        replica_count = min(self.prefix_replication, len(self._clients))
        offset = self._prefix_offsets.get(prefix, 0)
        self._prefix_offsets[prefix] = (offset + 1) % replica_count
        client_idx = (start + offset) % len(self._clients)
        return self._clients[client_idx]

    def add_observer(self, observer: LLMRequestObserver) -> None:
        super().add_observer(observer)
        for client in self._clients:
            client.add_observer(observer)

    def remove_observer(self, observer: LLMRequestObserver) -> None:
        super().remove_observer(observer)
        for client in self._clients:
            client.remove_observer(observer)

    async def generate(self, prompt: str | list[ChatMessage], **kwargs: Any) -> str:
        return await self._pick_client_for_prompt(prompt).generate(prompt, **kwargs)

    async def generate_raw(self, messages: list[dict], **kwargs: Any) -> str:
        """Zero-overhead generate with the same routing strategy as generate()."""
        client = self._pick_client_for_prompt(messages)
        return await client.generate_raw(messages, **kwargs)
