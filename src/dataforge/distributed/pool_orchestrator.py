# src/dataforge/distributed/pool_orchestrator.py
"""Orchestrator for the DataForge persistent worker pool.

Splits input into N shards, spawns N PersistentWorker processes, dispatches
jobs via file-based queue, waits for completion, and merges outputs.

Output files are named output_shard_{i}.jsonl so that the existing
merge_outputs() from distributed/shard.py (glob: *_shard_*.jsonl) works
without modification.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

from dataforge.distributed.pool_worker import _READY_FILE, _SHUTDOWN_SENTINEL, _worker_main
from dataforge.distributed.shard import merge_outputs, split_input

logger = logging.getLogger(__name__)

_JOB_TIMEOUT = 600.0    # seconds to wait for a job before declaring failure
_ALIVE_POLL = 0.5       # seconds between done-file polls


def _write_job_atomic(job_file: Path, job_data: dict[str, Any]) -> None:
    """Write job file atomically using a temp file + os.rename() (POSIX atomic)."""
    tmp = job_file.with_suffix(".job.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job_data, f)
    os.rename(tmp, job_file)  # atomic on POSIX


class PoolOrchestrator:
    """Manages N persistent worker processes for parallel shard processing."""

    def __init__(
        self,
        num_workers: int,
        work_dir: Path,
        spec: dict[str, Any],
    ) -> None:
        self.num_workers = num_workers
        self.work_dir = work_dir
        self.spec = spec
        self._processes: list[multiprocessing.Process] = []
        self._inbox_dirs: list[Path] = []

    def _setup_inbox(self) -> None:
        """Create (or clear) per-worker inbox directories."""
        self._inbox_dirs = []
        for i in range(self.num_workers):
            inbox = self.work_dir / "pool_inbox" / f"worker_{i}"
            # Clear stale files from any previous run
            if inbox.exists():
                for f in inbox.iterdir():
                    f.unlink(missing_ok=True)
            inbox.mkdir(parents=True, exist_ok=True)
            self._inbox_dirs.append(inbox)

    def _start_workers(self, base_urls: list[str]) -> None:
        self._processes = []
        for i, inbox in enumerate(self._inbox_dirs):
            # Each worker gets its own endpoint to avoid contention
            worker_spec = dict(self.spec)
            worker_spec["base_url"] = base_urls[i % len(base_urls)]
            p = multiprocessing.Process(
                target=_worker_main,
                args=(i, str(inbox), worker_spec),
                daemon=True,
                name=f"df-pool-worker-{i}",
            )
            p.start()
            self._processes.append(p)
        logger.info("Started %d persistent workers", self.num_workers)

    def _wait_for_workers_ready(self, timeout: float = 30.0) -> None:
        """Poll until all workers have written their .ready file.

        Replaces the old hardcoded time.sleep(1.5): workers signal readiness
        the moment they finish initialization, so dispatch happens immediately
        rather than after a fixed worst-case delay.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            all_ready = all(
                (inbox / _READY_FILE).exists() for inbox in self._inbox_dirs
            )
            if all_ready:
                logger.info("All %d workers ready", self.num_workers)
                return
            time.sleep(0.05)
        # If timeout hit, log warning but continue (workers may still process)
        ready_count = sum(1 for inbox in self._inbox_dirs if (inbox / _READY_FILE).exists())
        logger.warning(
            "Readiness timeout after %.1fs: %d/%d workers ready — proceeding anyway",
            timeout, ready_count, self.num_workers,
        )

    def _stop_workers(self) -> None:
        """Send shutdown sentinel to all workers, then join or terminate."""
        for i, inbox in enumerate(self._inbox_dirs):
            sentinel_file = inbox / f"shutdown_{i}.job"
            try:
                _write_job_atomic(sentinel_file, {"cmd": _SHUTDOWN_SENTINEL})
            except OSError:
                pass
        # Give workers 5s to exit gracefully, then terminate
        deadline = time.monotonic() + 5.0
        for p in self._processes:
            remaining = max(0.0, deadline - time.monotonic())
            p.join(timeout=remaining)
            if p.is_alive():
                p.terminate()

    def run(
        self,
        dataset: str,
        base_urls: list[str],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Split dataset, dispatch jobs, collect results, merge outputs.

        This is a synchronous method (uses time.sleep + multiprocessing).
        Call via asyncio.to_thread() from async contexts.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = str(self.work_dir / "shards")

        self._setup_inbox()

        # Start workers BEFORE splitting shards so boot time overlaps with I/O
        t0 = time.monotonic()
        self._start_workers(base_urls)

        # Split input into N shards (workers boot in parallel)
        shard_paths = split_input(dataset, num_shards=self.num_workers, output_dir=shard_dir)

        # Wait for all workers to signal readiness (replaces hardcoded 1.5s sleep)
        self._wait_for_workers_ready(timeout=30.0)

        # Dispatch one job per worker
        # Output files named output_shard_{i}.jsonl to match merge_outputs glob (*_shard_*.jsonl)
        job_ids: list[str] = []
        for i, shard_path in enumerate(shard_paths):
            job_id = f"job_{i}"
            output_path = str(Path(shard_dir) / f"output_shard_{i}.jsonl")
            job_data = {
                "shard_path": shard_path,
                "output_path": output_path,
                "concurrency": self.spec.get("concurrency", 50),
                "model": self.spec["model"],
                "base_url": base_urls[i % len(base_urls)],
                "api_key": self.spec.get("api_key", "EMPTY"),
                "depth": self.spec.get("depth", 1),
                "use_system_prompt": self.spec.get("use_system_prompt", True),
            }
            _write_job_atomic(self._inbox_dirs[i] / f"{job_id}.job", job_data)
            job_ids.append(job_id)

        # Wait for all jobs (with timeout + process liveness check)
        shard_results: list[dict[str, Any]] = []
        failed = 0
        for i, job_id in enumerate(job_ids):
            inbox = self._inbox_dirs[i]
            done_file = inbox / f"{job_id}.done"
            error_file = inbox / f"{job_id}.error"
            deadline = time.monotonic() + _JOB_TIMEOUT
            outcome: dict[str, Any] | None = None

            while time.monotonic() < deadline:
                if done_file.exists():
                    try:
                        with open(done_file, encoding="utf-8") as f:
                            outcome = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        pass
                    break
                if error_file.exists():
                    try:
                        with open(error_file, encoding="utf-8") as f:
                            err = json.load(f)
                        logger.error("Worker %d job %s failed: %s", i, job_id, err.get("error"))
                    except (json.JSONDecodeError, OSError):
                        pass
                    failed += 1
                    break
                # Detect silent worker crash (no .done or .error ever written)
                if not self._processes[i].is_alive():
                    logger.error("Worker %d process died unexpectedly for job %s", i, job_id)
                    failed += 1
                    break
                time.sleep(_ALIVE_POLL)
            else:
                logger.error("Worker %d job %s timed out after %ds", i, job_id, int(_JOB_TIMEOUT))
                failed += 1

            if outcome is not None:
                shard_results.append(outcome)

        self._stop_workers()

        elapsed = time.monotonic() - t0

        # Merge shard outputs (uses glob *_shard_*.jsonl, matches output_shard_{i}.jsonl)
        merged_path = str(output_dir / "merged_output.jsonl")
        total_records = merge_outputs(shard_dir, merged_path, dedup=True)

        return {
            "method": "dataforge_pool",
            "num_workers": self.num_workers,
            "total_records": total_records,
            "completed": total_records,
            "failed_workers": failed,
            "elapsed_seconds": elapsed,
            "records_per_second": total_records / elapsed if elapsed > 0 else 0.0,
            "records_per_minute": total_records / elapsed * 60 if elapsed > 0 else 0.0,
        }
