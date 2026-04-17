import types
from pathlib import Path

from experiments.exp1_throughput import run_throughput as rt


class DummyClient:
    def __init__(self) -> None:
        self.observers = []

    def add_observer(self, observer) -> None:
        self.observers.append(observer)


class DummyResult:
    total_records = 3
    completed = 3
    rejected = 0
    failed = 0
    elapsed_seconds = 1.5
    records_per_second = 2.0


async def test_run_dataforge_respects_observer_and_max_retries(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

        async def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return DummyResult()

    monkeypatch.setattr(rt, "Pipeline", FakePipeline)
    client = DummyClient()

    out = await rt.run_dataforge(
        client=client,
        dataset="in.jsonl",
        output_path="out.jsonl",
        concurrency=4,
        checkpoint_dir="ckpt",
        mode="burst",
        mutation_schedule="round_robin",
        mutation_batch_size=64,
        adaptive_concurrency=True,
        observer_enabled=False,
        max_retries=0,
        burst_window_size=32,
        prefix_aware_scheduling=True,
        prefix_affinity_striping=False,
    )

    assert out["method"] == "dataforge"
    assert captured["pipeline_kwargs"]["max_retries"] == 0
    assert captured["pipeline_kwargs"]["adaptive_concurrency"] is True
    assert captured["pipeline_kwargs"]["burst_window_size"] == 32
    assert captured["pipeline_kwargs"]["prefix_aware_scheduling"] is True
    assert captured["pipeline_kwargs"]["prefix_affinity_striping"] is False
    assert client.observers == []


async def test_run_baseline_uses_dataforge_workload_mode(monkeypatch):
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"method": "naive_async", "total_records": 0, "completed": 0, "failed": 0, "elapsed_seconds": 0.1, "records_per_minute": 0.0, "error_429_count": 0}

    monkeypatch.setattr(rt.NaiveAsyncBaseline, "run", fake_run)
    client = DummyClient()

    await rt.run_baseline(
        method="naive_asyncio",
        client=client,
        dataset="in.jsonl",
        output_path="out.jsonl",
        concurrency=8,
        backend="vllm",
        model="m",
        base_url="u",
        api_key=None,
        workload_mode="dataforge_strategy",
        mutation_schedule="batch",
        mutation_batch_size=64,
        max_retries=2,
    )

    assert "record_processor" in captured
    assert captured["record_processor"] is not None
    assert captured["max_retries"] == 2


async def test_run_baseline_rejects_dataforge_workload_for_threaded(monkeypatch):
    # Keep this guard so non-async threaded path is not silently treated as aligned.
    monkeypatch.setattr(
        rt,
        "ThreadedBaseline",
        types.SimpleNamespace(run=lambda **kwargs: {"ok": True}),
    )
    client = DummyClient()

    try:
        await rt.run_baseline(
            method="threaded",
            client=client,
            dataset="in.jsonl",
            output_path="out.jsonl",
            concurrency=4,
            backend="vllm",
            model="m",
            base_url="u",
            api_key=None,
            workload_mode="dataforge_strategy",
            mutation_schedule="random",
            mutation_batch_size=50,
            max_retries=0,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised


async def test_main_result_contains_runtime_metadata(monkeypatch, tmp_path):
    dataset = tmp_path / "seeds.jsonl"
    dataset.write_text('{"instruction":"hi"}\n', encoding="utf-8")
    captured = {}

    class FakeClient:
        rpm_limit = 321
        tpm_limit = 654321
        _disable_rate_limit = True

    async def fake_run_dataforge(**kwargs):
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

    def fake_save_result(result, output_path):
        captured["result"] = result
        captured["path"] = output_path

    monkeypatch.setattr(rt, "make_client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(rt, "run_dataforge", fake_run_dataforge)
    monkeypatch.setattr(rt, "save_result", fake_save_result)
    monkeypatch.setattr(rt, "print_summary", lambda result: None)

    args = types.SimpleNamespace(
        backend="vllm",
        dataset=str(dataset),
        concurrency=10,
        method="dataforge",
        output_dir=str(tmp_path / "out"),
        base_url="http://localhost:8000/v1",
        base_urls="http://localhost:8000/v1,http://localhost:8001/v1",
        cluster_routing="prefix_affinity",
        prefix_replication=3,
        api_key=None,
        model="m",
        mode="burst",
        mutation_schedule="batch",
        mutation_batch_size=64,
        workload_mode="dataforge_strategy",
        adaptive_concurrency=False,
        observer_enabled=False,
        max_retries=2,
        burst_window_size=32,
        prefix_aware_scheduling=True,
        disable_prefix_affinity_striping=False,
        use_system_prompt=False,
        scenario="ideal",
        failure_rate=0.1,
        rpm_limit=111,
        tpm_limit=222,
    )
    await rt.main(args)

    assert Path(captured["path"]).name.startswith("exp1_dataforge_vllm_")
    assert captured["result"]["disable_rate_limit"] is True
    assert captured["result"]["rpm_limit"] == 321
    assert captured["result"]["tpm_limit"] == 654321
    assert captured["result"]["base_urls"] == ["http://localhost:8000/v1", "http://localhost:8001/v1"]
    assert captured["result"]["cluster_routing"] == "prefix_affinity"
    assert captured["result"]["prefix_replication"] == 3
    assert captured["result"]["mode"] == "burst"
    assert captured["result"]["mutation_schedule"] == "batch"
    assert captured["result"]["mutation_batch_size"] == 64
    assert captured["result"]["burst_window_size"] == 32
    assert captured["result"]["prefix_aware_scheduling"] is True
    assert captured["result"]["prefix_affinity_striping"] is True
    assert "effective_records_per_minute" in captured["result"]
    assert "completion_rate" in captured["result"]


async def test_run_baseline_supports_observer_enabled(monkeypatch):
    client = DummyClient()
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "method": "naive_async",
            "total_records": 1,
            "completed": 1,
            "failed": 0,
            "elapsed_seconds": 1.0,
            "records_per_minute": 60.0,
            "error_429_count": 0,
        }

    monkeypatch.setattr(rt.NaiveAsyncBaseline, "run", fake_run)
    await rt.run_baseline(
        method="naive_asyncio",
        client=client,
        dataset="in.jsonl",
        output_path="out.jsonl",
        concurrency=8,
        backend="vllm",
        model="m",
        base_url="u",
        api_key=None,
        workload_mode="baseline_prompt",
        mutation_schedule="random",
        mutation_batch_size=50,
        max_retries=0,
        observer_enabled=True,
    )
    assert len(client.observers) == 1


def test_make_client_supports_vllm_cluster():
    client = rt.make_client(
        backend="vllm",
        model="m",
        base_url="http://localhost:8000/v1",
        base_urls=["http://localhost:8000/v1", "http://localhost:8001/v1"],
        cluster_routing="round_robin",
        prefix_replication=2,
        api_key=None,
    )

    assert client.__class__.__name__ == "vLLMClusterClient"
    assert client.base_urls == ["http://localhost:8000/v1", "http://localhost:8001/v1"]
