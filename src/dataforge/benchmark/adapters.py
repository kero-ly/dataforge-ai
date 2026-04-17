from __future__ import annotations

from dataforge.clients.base import BaseLLMClient


class LLMCandidateAdapter:
    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client

    async def generate(self, prompt: str) -> str:
        return await self.client.generate(prompt)


class LLMJudgeAdapter:
    def __init__(self, client: BaseLLMClient) -> None:
        self.client = client

    async def score(self, prompt: str) -> str:
        return await self.client.generate(prompt)
