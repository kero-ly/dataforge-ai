from dataforge.benchmark.adapters import LLMCandidateAdapter, LLMJudgeAdapter


class MockLLM:
    async def generate(self, prompt, **kwargs):  # noqa: ANN001
        return f"response:{prompt}"


async def test_benchmark_adapters_proxy_generate():
    client = MockLLM()
    candidate = LLMCandidateAdapter(client)  # type: ignore[arg-type]
    judge = LLMJudgeAdapter(client)  # type: ignore[arg-type]

    assert await candidate.generate("hello") == "response:hello"
    assert await judge.score("judge this") == "response:judge this"
