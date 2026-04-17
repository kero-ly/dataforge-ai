from __future__ import annotations

import argparse
import json

from experiments.exp2_api_effective import analyze_effective_api
from experiments.exp2_prefix_cache import analyze_prefix_cache


def test_analyze_effective_api_handles_single_and_multiple_runs(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "figures"
    results_dir.mkdir()
    output_dir.mkdir()
    sample_rows = [
        {
            "experiment": "exp2_api_effective",
            "provider": "deepseek",
            "method": "dataforge",
            "effective_records_per_minute": 12.0,
            "completion_rate": 1.0,
            "error_429_count": 0,
            "cost_per_completed_record": 0.01,
        },
        {
            "experiment": "exp2_api_effective",
            "provider": "deepseek",
            "method": "dataforge",
            "effective_records_per_minute": 14.0,
            "completion_rate": 0.9,
            "error_429_count": 1,
            "cost_per_completed_record": 0.02,
        },
        {
            "experiment": "exp2_api_effective",
            "provider": "deepseek",
            "method": "naive_asyncio",
            "effective_records_per_minute": 10.0,
            "completion_rate": 0.8,
            "error_429_count": 3,
            "cost_per_completed_record": 0.03,
        },
    ]
    for index, row in enumerate(sample_rows):
        (results_dir / f"exp2_api_effective_{index}.json").write_text(json.dumps(row), encoding="utf-8")

    analyze_effective_api.main(
        argparse.Namespace(
            results_dir=str(results_dir),
            output_dir=str(output_dir),
            provider="deepseek",
        )
    )
    assert (output_dir / "exp2_effective_api_table.tex").exists()
    assert (output_dir / "exp2_effective_throughput.pdf").exists()


def test_analyze_prefix_cache_handles_runs(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "figures"
    results_dir.mkdir()
    output_dir.mkdir()
    for index, schedule in enumerate(["random", "prefix_affinity_striped"]):
        row = {
            "experiment": "exp2_prefix_cache",
            "schedule_name": schedule,
            "dataset_size": 10000,
            "records_per_minute": 3000 + index * 100,
            "prefix_cache_hit_rate": 0.7 + 0.1 * index,
            "ttft_seconds": {"mean": 0.05 + 0.01 * index},
        }
        (results_dir / f"exp2_prefix_cache_{index}.json").write_text(json.dumps(row), encoding="utf-8")
    analyze_prefix_cache.main(
        argparse.Namespace(
            results_dir=str(results_dir),
            output_dir=str(output_dir),
            dataset_size=10000,
        )
    )
    assert (output_dir / "exp2_prefix_cache_table.tex").exists()
    assert (output_dir / "exp2_prefix_cache_triptych.pdf").exists()
