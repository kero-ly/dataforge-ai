from __future__ import annotations

from dataforge.clients.openai_client import OpenAIClient


class BailianClient(OpenAIClient):
    """LLM client for Alibaba Cloud Model Studio (Bailian) OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
