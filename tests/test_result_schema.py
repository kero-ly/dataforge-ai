from __future__ import annotations

from experiments.common.result_schema import (
    aggregate_numeric_results,
    standardize_result_schema,
    summarize_output_tokens,
)


def test_standardize_result_schema_derives_effective_and_cost_fields():
    result = standardize_result_schema(
        {
            "method": "dataforge",
            "backend": "deepseek",
            "dataset": "seeds.jsonl",
            "dataset_size": 10,
            "total_records": 10,
            "concurrency": 5,
            "completed": 8,
            "failed": 2,
            "elapsed_seconds": 30.0,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "error_429_count": 1,
        },
        provider="deepseek",
        prompt_price_per_1m=1.0,
        completion_price_per_1m=2.0,
    )

    assert result["completion_rate"] == 0.8
    assert result["effective_records_per_minute"] == 16.0
    assert result["estimated_cost_usd"] == 0.002
    assert result["cost_per_completed_record"] == 0.00025
    assert result["wasted_requests_per_completed_record"] == 0.25


def test_summarize_output_tokens_uses_framework_fields():
    prompt_tokens, completion_tokens = summarize_output_tokens(
        [
            {"instruction": "hello world", "generation": "answer"},
            {"query": "another prompt", "response": "completion"},
        ]
    )
    assert prompt_tokens > 0
    assert completion_tokens > 0


def test_aggregate_numeric_results_handles_single_and_multi_runs():
    rows = aggregate_numeric_results(
        [
            {"method": "dataforge", "provider": "deepseek", "effective_records_per_minute": 10.0},
            {"method": "dataforge", "provider": "deepseek", "effective_records_per_minute": 14.0},
            {"method": "naive_asyncio", "provider": "deepseek", "effective_records_per_minute": 11.0},
        ],
        group_keys=["provider", "method"],
        numeric_keys=["effective_records_per_minute"],
    )
    lookup = {(row["provider"], row["method"]): row for row in rows}
    assert lookup[("deepseek", "dataforge")]["effective_records_per_minute_mean"] == 12.0
    assert lookup[("deepseek", "naive_asyncio")]["effective_records_per_minute_std"] == 0.0
