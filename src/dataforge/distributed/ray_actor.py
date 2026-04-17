"""DataForge Ray Actor distributed backend.

Wraps DataForge Pipeline inside a Ray remote Actor so that:
  - Each Actor's asyncio event loop persists across shard jobs (no cold start)
  - HTTP connection pool inside vLLMClient is never destroyed
  - All DataForge features work: RetryEngine, EvolInstruct, LLMJudge, prefix scheduling

Architecture::

    RayOrchestrator.run(input_path, output_path, ...)
      ├── ray.init(address=ray_address)     # local or existing cluster
      ├── split_input() → N shard files
      ├── DataForgeActor × N  (Ray schedules across nodes)
      ├── actor.process_shard.remote() × N  (all parallel)
      ├── ray.get([...])
      └── merge_outputs()

Usage::

    from dataforge.distributed.ray_actor import RayOrchestrator

    result = RayOrchestrator.run(
        input_path="seeds_1k.jsonl",
        output_path="output.jsonl",
        model="Qwen/Qwen2.5-7B-Instruct",
        base_urls=["http://localhost:8100/v1", "http://localhost:8101/v1",
                   "http://localhost:8102/v1", "http://localhost:8103/v1"],
        # Replace with your actual vLLM server URLs
        num_actors=4,
        concurrency_per_actor=50,
    )
    # result["records_per_minute"] -> float
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _clear_metrics(pipeline: Any) -> None:
    """Remove stale MetricsCollector so a fresh one is created on next run."""
    try:
        from dataforge.metrics import MetricsCollector
        pipeline.hooks = [h for h in pipeline.hooks if not isinstance(h, MetricsCollector)]
    except ImportError:
        pass


def _make_actor_class():
    """Return the DataForgeActor class (defined inside Ray context)."""
    try:
        import ray
    except ImportError as e:
        raise ImportError(
            "ray is required. Install with: pip install 'ray[default]'"
        ) from e

    @ray.remote
    class DataForgeActor:
        """Ray Actor wrapping a DataForge Pipeline.

        Created once per distributed job; the Pipeline instance (and its
        underlying vLLMClient / httpx connection pool) persists for the
        Actor's entire lifetime.  Ray manages a single asyncio event loop
        per Actor — no event-loop teardown between shard jobs.
        """

        def __init__(
            self,
            actor_id: int,
            model: str,
            base_url: str,
            concurrency: int = 50,
            use_evaluator: bool = False,
            eval_concurrency: int | None = None,
            judge_threshold: float = 1.0,
            max_retries: int = 3,
            depth: int = 1,
        ) -> None:
            import sys
            from pathlib import Path as _Path

            _root = _Path(__file__).resolve().parents[3]
            for _p in (str(_root / "src"), str(_root)):
                if _p not in sys.path:
                    sys.path.insert(0, _p)

            from dataforge.clients.vllm_client import vLLMClient
            from dataforge.pipeline import Pipeline
            from dataforge.strategies.evol_instruct import EvolInstruct

            client = vLLMClient(
                model=model,
                base_url=base_url,
                rpm_limit=999_999,
                tpm_limit=0,
            )
            strategy = EvolInstruct(llm=client, depth=depth, use_system_prompt=True)

            evaluators = []
            if use_evaluator:
                from dataforge.evaluators.llm_judge import LLMJudge
                evaluators = [LLMJudge(
                    llm=client,
                    threshold=judge_threshold,
                    eval_max_tokens=5,
                )]

            pipeline_kwargs: dict[str, Any] = dict(
                strategy=strategy,
                evaluators=evaluators,
                checkpoint_disabled=True,
                max_retries=max_retries,
                retry_base_delay=0.0,
            )
            if eval_concurrency is not None:
                pipeline_kwargs["eval_concurrency"] = eval_concurrency

            self.pipeline = Pipeline(**pipeline_kwargs)
            self.concurrency = concurrency
            self.actor_id = actor_id
            self._model = model
            self._base_url = base_url

        async def warmup(self) -> str:
            """Send a minimal request to warm up HTTP connection + vLLM KV cache."""
            import openai
            try:
                client = openai.AsyncOpenAI(
                    base_url=self._base_url, api_key="EMPTY", timeout=30,
                )
                await client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                return f"warmup_failed:{exc}"
            return "ok"

        async def process_shard(
            self,
            shard_path: str,
            output_path: str,
        ) -> dict[str, Any]:
            """Process one shard file and write completed records to output_path."""
            _clear_metrics(self.pipeline)
            t0 = time.monotonic()
            result = await self.pipeline.run(
                input_path=shard_path,
                output_path=output_path,
                concurrency=self.concurrency,
                mode="burst",
                show_progress=False,
            )
            elapsed = time.monotonic() - t0
            return {
                "actor_id": self.actor_id,
                "shard_path": shard_path,
                "total_records": result.total_records,
                "completed": result.completed,
                "failed": result.failed,
                "elapsed_seconds": elapsed,
                "records_per_minute": result.total_records / elapsed * 60 if elapsed > 0 else 0.0,
            }

    return DataForgeActor


class RayOrchestrator:
    """Orchestrates N DataForgeActors across a Ray cluster."""

    @staticmethod
    def run(
        input_path: str,
        output_path: str,
        *,
        model: str,
        base_urls: list[str],
        num_actors: int,
        concurrency_per_actor: int = 50,
        ray_address: str | None = None,
        use_evaluator: bool = False,
        eval_concurrency: int | None = None,
        judge_threshold: float = 1.0,
        max_retries: int = 3,
        depth: int = 1,
        work_dir: str | None = None,
        warmup: bool = True,
    ) -> dict[str, Any]:
        """Run DataForge pipeline across N Ray Actors.

        Args:
            input_path: Path to input JSONL file.
            output_path: Path where merged output will be written.
            model: LLM model name (passed to every Actor).
            base_urls: List of vLLM base URLs.  Actor i uses
                base_urls[i % len(base_urls)], so you can have fewer
                URLs than actors (round-robin) or one URL per actor.
            num_actors: Number of parallel Actors.
            concurrency_per_actor: asyncio Semaphore size inside each Actor.
            ray_address: Ray cluster address.  None = local Ray instance.
                "auto" = connect to existing cluster.
                "ray://host:10001" = remote cluster.
            use_evaluator: Whether to attach a LLMJudge evaluator.
            work_dir: Temporary directory for shard files.  Auto-created
                under /tmp if not specified.
            warmup: Send a warmup request from each Actor before timing.

        Returns:
            Metrics dict with keys: method, num_actors, total_records,
            completed, elapsed_seconds, records_per_minute, shard_results.
        """
        try:
            import ray
        except ImportError as e:
            raise ImportError(
                "ray is required. Install with: pip install 'ray[default]'"
            ) from e

        from dataforge.distributed.shard import merge_outputs, split_input

        if not ray.is_initialized():
            # Pass repo paths via runtime_env so Actor worker processes can
            # import dataforge even without a system-wide pip install.
            import os
            existing = os.environ.get("PYTHONPATH", "")
            repo_paths = f"{_REPO_ROOT / 'src'}:{_REPO_ROOT}"
            pythonpath = f"{repo_paths}:{existing}" if existing else repo_paths
            ray.init(
                address=ray_address,
                ignore_reinit_error=True,
                logging_level=logging.WARNING,
                runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
            )

        # Prepare shard directory
        _work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="df_ray_"))
        shard_dir = _work / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        # Split input
        shard_paths = split_input(input_path, num_actors, str(shard_dir))
        output_paths = [
            str(shard_dir / f"output_shard_{i}.jsonl")
            for i in range(num_actors)
        ]

        # Create Actors
        DataForgeActor = _make_actor_class()
        actors = [
            DataForgeActor.remote(
                i,
                model,
                base_urls[i % len(base_urls)],
                concurrency_per_actor,
                use_evaluator,
                eval_concurrency,
                judge_threshold,
                max_retries,
                depth,
            )
            for i in range(num_actors)
        ]

        # Warmup
        if warmup:
            warmup_results = ray.get([a.warmup.remote() for a in actors])
            failed_warmups = [r for r in warmup_results if r != "ok"]
            if failed_warmups:
                logger.warning("Some Actor warmups failed: %s", failed_warmups)

        # Dispatch all shards in parallel
        t0 = time.monotonic()
        futures = [
            a.process_shard.remote(sp, op)
            for a, sp, op in zip(actors, shard_paths, output_paths)
        ]
        shard_results: list[dict[str, Any]] = ray.get(futures)
        elapsed = time.monotonic() - t0

        # Merge shard outputs
        merge_outputs(str(shard_dir), output_path)

        # Aggregate
        total_input = sum(r["total_records"] for r in shard_results)
        total_completed = sum(r["completed"] for r in shard_results)

        return {
            "method": "DataForge_RayActor",
            "num_actors": num_actors,
            "total_records": total_input,
            "completed": total_completed,
            "completion_rate": total_completed / total_input if total_input > 0 else 0.0,
            "elapsed_seconds": elapsed,
            "records_per_minute": total_input / elapsed * 60 if elapsed > 0 else 0.0,
            "shard_results": shard_results,
        }
