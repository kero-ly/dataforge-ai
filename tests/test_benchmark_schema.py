from dataforge.benchmark.schema import BenchmarkCase, BenchmarkCaseResult, BenchmarkRunSummary, BenchmarkTaskSummary


def test_benchmark_schema_models_round_trip():
    case = BenchmarkCase(id="c1", category="math", prompt="2+2?")
    result = BenchmarkCaseResult(
        case_id="c1",
        category="math",
        prompt="2+2?",
        response="4",
        raw_score=10.0,
        normalized_score=100.0,
        passed=True,
    )
    task_summary = BenchmarkTaskSummary(
        task="mt_bench_lite_v1",
        category_scores={"math": 100.0},
        overall_score=100.0,
        success_rate=1.0,
        num_cases=1,
        num_errors=0,
    )
    run_summary = BenchmarkRunSummary(
        benchmark="demo",
        candidate_name="mock",
        task_summaries=[task_summary],
        overall_score=100.0,
        weighted_scores={"mt_bench_lite_v1": 100.0},
    )

    assert case.model_dump()["prompt"] == "2+2?"
    assert result.model_dump()["normalized_score"] == 100.0
    assert run_summary.model_dump()["candidate_name"] == "mock"
