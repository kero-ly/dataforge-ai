import tempfile
from pathlib import Path

from dataforge.benchmark.schema import BenchmarkRunSummary, BenchmarkTaskSummary


class MockBenchmarkRunner:
    async def run(self):
        summary = BenchmarkRunSummary(
            benchmark="demo",
            candidate_name="mock",
            task_summaries=[
                BenchmarkTaskSummary(
                    task="mt_bench_lite_v1",
                    overall_score=80.0,
                    success_rate=1.0,
                    num_cases=1,
                    num_errors=0,
                )
            ],
            overall_score=80.0,
            weighted_scores={"mt_bench_lite_v1": 80.0},
        )
        return Path("/tmp"), summary


def test_benchmark_dry_run(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "benchmark.yaml"
        config_path.write_text(
            "\n".join(
                [
                    'kind: "benchmark"',
                    'name: "demo"',
                    "candidate:",
                    '  provider: "vllm"',
                    '  model: "mock"',
                    '  base_url: "http://localhost"',
                    "tasks:",
                    '  - name: "mt_bench_lite_v1"',
                    "output:",
                    f'  dir: "{tmpdir}"',
                ]
            ),
            encoding="utf-8",
        )
        from dataforge.cli import _benchmark

        _benchmark(str(config_path), dry_run=True)

        assert "Benchmark config is valid" in capsys.readouterr().out
