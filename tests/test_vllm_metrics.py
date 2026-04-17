from __future__ import annotations

from experiments.common.vllm_metrics import (
    diff_histogram,
    metrics_url,
    parse_prometheus_text,
    summarize_vllm_run,
)


PROM_TEXT_BEFORE = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries.
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:prefix_cache_hits_total{engine="0"} 40
vllm:prompt_tokens_total{engine="0"} 100
vllm:prompt_tokens_cached_total{engine="0"} 35
vllm:request_success_total{engine="0",finished_reason="stop"} 10
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1"} 3
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.2"} 8
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 10
vllm:time_to_first_token_seconds_count{engine="0"} 10
vllm:time_to_first_token_seconds_sum{engine="0"} 1.4
vllm:request_prefill_time_seconds_bucket{engine="0",le="0.1"} 2
vllm:request_prefill_time_seconds_bucket{engine="0",le="0.2"} 9
vllm:request_prefill_time_seconds_bucket{engine="0",le="+Inf"} 10
vllm:request_prefill_time_seconds_count{engine="0"} 10
vllm:request_prefill_time_seconds_sum{engine="0"} 1.3
"""

PROM_TEXT_AFTER = """
vllm:prefix_cache_queries_total{engine="0"} 160
vllm:prefix_cache_hits_total{engine="0"} 100
vllm:prompt_tokens_total{engine="0"} 170
vllm:prompt_tokens_cached_total{engine="0"} 90
vllm:request_success_total{engine="0",finished_reason="stop"} 22
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1"} 7
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.2"} 18
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 22
vllm:time_to_first_token_seconds_count{engine="0"} 22
vllm:time_to_first_token_seconds_sum{engine="0"} 2.96
vllm:request_prefill_time_seconds_bucket{engine="0",le="0.1"} 3
vllm:request_prefill_time_seconds_bucket{engine="0",le="0.2"} 18
vllm:request_prefill_time_seconds_bucket{engine="0",le="+Inf"} 22
vllm:request_prefill_time_seconds_count{engine="0"} 22
vllm:request_prefill_time_seconds_sum{engine="0"} 2.74
"""


def test_metrics_url_rewrites_openai_path():
    assert metrics_url("http://localhost:8000/v1") == "http://localhost:8000/metrics"
    assert metrics_url("http://localhost:8000") == "http://localhost:8000/metrics"


def test_diff_histogram_extracts_quantiles():
    before = parse_prometheus_text(PROM_TEXT_BEFORE)
    after = parse_prometheus_text(PROM_TEXT_AFTER)

    summary = diff_histogram(before, after, "vllm:time_to_first_token_seconds")

    assert summary.count == 12
    assert summary.mean == 0.13
    assert summary.p50 == 0.2
    assert summary.p95 == 0.2


def test_summarize_vllm_run_computes_cache_and_latency_deltas():
    before = parse_prometheus_text(PROM_TEXT_BEFORE)
    after = parse_prometheus_text(PROM_TEXT_AFTER)

    summary = summarize_vllm_run(before, after)

    assert summary["prefix_cache_queries"] == 60
    assert summary["prefix_cache_hits"] == 60
    assert summary["prefix_cache_hit_rate"] == 1.0
    assert summary["prompt_tokens"] == 70
    assert summary["prompt_tokens_cached"] == 55
    assert round(summary["cached_prompt_token_ratio"], 6) == round(55 / 70, 6)
    assert summary["request_success_count"] == 12
    assert summary["ttft_seconds"]["p50"] == 0.2
    assert summary["prefill_seconds"]["p95"] == 0.2
