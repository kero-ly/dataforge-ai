from dataforge.benchmark.runner import BenchmarkRunner
from dataforge.config.benchmark_schema import BenchmarkConfig, BenchmarkOutputConfig, BenchmarkTaskConfig
from dataforge.config.schema import LLMConfig


class MockClient:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def generate(self, prompt, **kwargs):  # noqa: ANN001
        if self.mode == "judge":
            if "Only output the number" in prompt:
                return "8"
            if "safe_comply" in prompt:
                return "safe_comply"
            return "refuse"
        if "JSON object" in str(prompt):
            return '{"name": "demo", "score": 5}'
        if "bullet list" in str(prompt):
            return "- tip one\n- tip two\n- tip three"
        if "apple" in str(prompt):
            return "apple is great"
        return "safe helpful response"


async def test_benchmark_runner_generates_report():
    config = BenchmarkConfig(
        name="demo-benchmark",
        candidate=LLMConfig(provider="vllm", model="mock", base_url="http://localhost"),
        tasks=[
            BenchmarkTaskConfig(name="mt_bench_lite_v1"),
            BenchmarkTaskConfig(name="if_eval_lite_v1"),
            BenchmarkTaskConfig(name="safety_lite_v1"),
        ],
        output=BenchmarkOutputConfig(dir="/tmp"),
    )
    object.__setattr__(config, "_candidate_client", MockClient("candidate"))
    object.__setattr__(config, "_judge_client", MockClient("judge"))

    output_dir, summary = await BenchmarkRunner(config).run()

    assert summary.overall_score is not None
    assert output_dir.exists()
    assert (output_dir / "summary.json").exists()
