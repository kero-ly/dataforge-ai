from __future__ import annotations

import types

from experiments.exp2_prefix_cache import run_prefix_cache as rpc


async def test_run_single_attaches_vllm_metrics_and_schedule(monkeypatch, tmp_path):
    dataset = tmp_path / "seeds.jsonl"
    dataset.write_text('{"instruction":"hi"}\n', encoding="utf-8")
    captured = {}

    class FakeSnapshot:
        pass

    snapshots = [FakeSnapshot(), FakeSnapshot()]

    monkeypatch.setattr(rpc, "capture_vllm_metrics", lambda base_urls: snapshots.pop(0))
    monkeypatch.setattr(rpc, "make_client", lambda **kwargs: types.SimpleNamespace(**kwargs))

    async def fake_run_dataforge(**kwargs):
        captured.update(kwargs)
        return {
            "method": "dataforge",
            "total_records": 1,
            "completed": 1,
            "rejected": 0,
            "failed": 0,
            "elapsed_seconds": 1.0,
            "records_per_second": 1.0,
            "records_per_minute": 60.0,
        }

    monkeypatch.setattr(rpc, "run_dataforge", fake_run_dataforge)
    monkeypatch.setattr(
        rpc,
        "summarize_vllm_run",
        lambda before, after: {"prefix_cache_hit_rate": 0.75, "ttft_seconds": {"p50": 0.1}},
    )

    args = types.SimpleNamespace(
        dataset=str(dataset),
        output_dir=str(tmp_path / "out"),
        base_url="http://localhost:8000/v1",
        base_urls="http://localhost:8000/v1,http://localhost:8001/v1",
        model="m",
        concurrency=8,
        mutation_batch_size=64,
        burst_window_size=0,
        prefix_replication=2,
        rpm_limit=None,
        tpm_limit=None,
    )

    result = await rpc.run_single(args, "prefix_affinity")

    assert captured["prefix_aware_scheduling"] is True
    assert captured["mutation_schedule"] == "batch"
    assert captured["prefix_affinity_striping"] is False
    assert result["cluster_routing"] == "prefix_affinity"
    assert result["vllm_metrics"]["prefix_cache_hit_rate"] == 0.75
    assert result["prefix_cache_hit_rate"] == 0.75


async def test_run_single_supports_prefix_affinity_striped(monkeypatch, tmp_path):
    dataset = tmp_path / "seeds.jsonl"
    dataset.write_text('{"instruction":"hi"}\n', encoding="utf-8")

    class FakeSnapshot:
        pass

    snapshots = [FakeSnapshot(), FakeSnapshot()]

    monkeypatch.setattr(rpc, "capture_vllm_metrics", lambda base_urls: snapshots.pop(0))
    monkeypatch.setattr(rpc, "make_client", lambda **kwargs: types.SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        rpc,
        "summarize_vllm_run",
        lambda before, after: {"prefix_cache_hit_rate": 0.9},
    )

    captured = {}

    async def fake_run_dataforge(**kwargs):
        captured.update(kwargs)
        return {
            "method": "dataforge",
            "total_records": 1,
            "completed": 1,
            "rejected": 0,
            "failed": 0,
            "elapsed_seconds": 1.0,
            "records_per_second": 1.0,
            "records_per_minute": 60.0,
        }

    monkeypatch.setattr(rpc, "run_dataforge", fake_run_dataforge)

    args = types.SimpleNamespace(
        dataset=str(dataset),
        output_dir=str(tmp_path / "out"),
        base_url="http://localhost:8000/v1",
        base_urls=None,
        model="m",
        concurrency=8,
        mutation_batch_size=64,
        burst_window_size=0,
        prefix_replication=2,
        rpm_limit=None,
        tpm_limit=None,
    )

    result = await rpc.run_single(args, "prefix_affinity_striped")
    assert captured["prefix_affinity_striping"] is True
    assert result["schedule_name"] == "prefix_affinity_striped"
