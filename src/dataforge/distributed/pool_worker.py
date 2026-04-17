# src/dataforge/distributed/pool_worker.py
"""Persistent worker process for the DataForge pool-based distributed mode.

Each PersistentWorker stays alive across multiple shard jobs, maintaining a
single Pipeline instance (and its underlying HTTP connection pool) to avoid
per-subprocess cold-start and connection-rebuild overhead.

Job protocol (file-based, no Redis/socket deps):
  pool_inbox/worker_{i}/{job_id}.job      JSON config written atomically via rename
  pool_inbox/worker_{i}/{job_id}.ready    Empty file written by worker when init done
  pool_inbox/worker_{i}/{job_id}.done     JSON result written by worker
  pool_inbox/worker_{i}/{job_id}.error    JSON error written by worker on failure
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.05   # seconds between inbox polls (halved for lower latency)
_SHUTDOWN_SENTINEL = "__SHUTDOWN__"
_READY_FILE = ".ready"


def _build_pipeline_from_spec(spec: dict[str, Any]):
    """Build a Pipeline from a minimal job spec dict."""
    import sys
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    for candidate in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    from dataforge.clients.vllm_client import vLLMClient
    from dataforge.pipeline import Pipeline
    from dataforge.strategies.evol_instruct import EvolInstruct

    client = vLLMClient(
        model=spec["model"],
        base_url=spec["base_url"],
        rpm_limit=999999,
        tpm_limit=0,
    )
    strategy = EvolInstruct(
        llm=client,
        depth=spec.get("depth", 1),
        use_system_prompt=spec.get("use_system_prompt", True),
    )
    pipeline = Pipeline(
        strategy=strategy,
        checkpoint_disabled=True,  # persistent workers never checkpoint
        max_retries=3,
        retry_base_delay=0.0,
    )
    return pipeline


def _clear_metrics(pipeline: Any) -> None:
    """Remove stale MetricsCollector so a fresh one is created on next run."""
    try:
        from dataforge.metrics import MetricsCollector
        pipeline.hooks = [h for h in pipeline.hooks if not isinstance(h, MetricsCollector)]
    except ImportError:
        pass


async def _worker_main_async(worker_id: int, inbox_dir: str, spec: dict[str, Any]) -> None:
    """Persistent async worker loop.

    Runs under a single asyncio event loop for the worker's entire lifetime.
    The httpx AsyncClient (inside vLLMClient) is created once and reused
    across all jobs — no connection pool teardown between shard batches.
    """
    inbox = Path(inbox_dir)
    inbox.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline_from_spec(spec)

    # Signal readiness so orchestrator can dispatch without a fixed sleep
    ready_file = inbox / _READY_FILE
    ready_file.touch()
    logger.warning("Worker %d ready, polling %s", worker_id, inbox)

    while True:
        job_files = sorted(f for f in inbox.glob("*.job") if f.name != _READY_FILE)
        if not job_files:
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        job_file = job_files[0]
        try:
            with open(job_file, encoding="utf-8") as f:
                job = json.load(f)
        except (json.JSONDecodeError, OSError):
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        if job.get("cmd") == _SHUTDOWN_SENTINEL:
            logger.warning("Worker %d received shutdown sentinel", worker_id)
            job_file.unlink(missing_ok=True)
            break

        job_id = job_file.stem
        done_file = inbox / f"{job_id}.done"
        error_file = inbox / f"{job_id}.error"

        # Remove job file first so we don't re-process it on crash restart
        job_file.unlink(missing_ok=True)

        try:
            _clear_metrics(pipeline)
            t0 = time.monotonic()
            # Await directly — no new event loop, connection pool persists
            await pipeline.run(
                input_path=job["shard_path"],
                output_path=job["output_path"],
                concurrency=job.get("concurrency", 50),
                mode="burst",
            )
            elapsed = time.monotonic() - t0
            total_records = 0
            try:
                with open(job["output_path"], encoding="utf-8") as f:
                    total_records = sum(1 for line in f if line.strip())
            except OSError:
                pass
            done_data = {"records": total_records, "elapsed_seconds": elapsed}
            with open(done_file, "w", encoding="utf-8") as f:
                json.dump(done_data, f)
            logger.warning(
                "Worker %d job %s: %d records in %.1fs",
                worker_id, job_id, total_records, elapsed,
            )
        except BaseException as exc:  # noqa: BLE001
            logger.error("Worker %d job %s failed: %s", worker_id, job_id, exc, exc_info=True)
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump({"error": str(exc)}, f)


def _worker_main(worker_id: int, inbox_dir: str, spec: dict[str, Any]) -> None:
    """Entry point for a persistent worker process.

    Runs in a separate process (via multiprocessing.Process). A single
    asyncio.run() call covers the entire worker lifetime — the event loop
    and HTTP connection pool are never torn down between jobs.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format=f"[worker-{worker_id}] %(levelname)s %(message)s",
    )
    asyncio.run(_worker_main_async(worker_id, inbox_dir, spec))
