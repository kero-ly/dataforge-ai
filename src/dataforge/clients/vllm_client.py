# src/dataforge/clients/vllm_client.py
from __future__ import annotations

from dataforge.clients.openai_client import OpenAIClient


class vLLMClient(OpenAIClient):
    """LLM client for local vLLM / Ollama / SGLang servers (OpenAI-compatible API).

    vLLM servers expose an OpenAI-compatible REST API, so this client inherits
    all behavior from OpenAIClient. Key differences:
    - Defaults base_url to the standard vLLM endpoint (localhost:8000)
    - Does not require a real API key (vLLM does not validate them)
    - Higher default RPM/TPM limits since local servers handle more throughput

    The generate() method is fully inherited from OpenAIClient.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        rpm_limit: int = 5000,
        tpm_limit: int = 10_000_000,
        disable_rate_limit: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key="not-needed",  # vLLM does not validate API keys
            base_url=base_url,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            disable_rate_limit=disable_rate_limit,
            **kwargs,
        )
