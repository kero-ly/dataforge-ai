# src/dataforge/distributed/worker.py
"""Distributed worker process for the Coordinator-Worker mode.

Each worker:
1. Pulls records from a Redis task queue (BRPOP)
2. Processes them through the Pipeline's strategy + evaluators
3. Publishes results back via Redis Pub/Sub
4. Commits completed records to the shared Redis checkpoint

Usage::

    worker = DistributedWorker(pipeline, config)
    await worker.run()
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import time
from typing import Any

from dataforge.engine.redis_checkpoint import RedisCheckpointManager
from dataforge.metrics import PipelineResult
from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord, RecordStatus

logger = logging.getLogger(__name__)


def _import_redis():
    try:
        import redis.asyncio as aioredis
        return aioredis
    except ImportError:
        raise ImportError(
            "redis is required for distributed mode. "
            "Install it with: pip install 'dataforge[distributed]'"
        ) from None


class DistributedWorker:
    """A distributed worker that pulls tasks from Redis and processes them.

    Reuses the Pipeline's ``process_record()`` method for the actual
    strategy + evaluator chain.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        redis_url: str = "redis://localhost:6379",
        run_id: str = "default",
        queue_name: str = "dataforge:tasks",
        result_channel: str = "dataforge:results",
        concurrency: int = 50,
        worker_id: str = "worker-0",
    ) -> None:
        self._pipeline = pipeline
        self._redis_url = redis_url
        self._run_id = run_id
        self._queue_name = queue_name
        self._result_channel = result_channel
        self._concurrency = concurrency
        self._worker_id = worker_id
        self._checkpoint = RedisCheckpointManager(redis_url, run_id=run_id)
        self._shutdown = False

    async def run(self) -> PipelineResult:
        """Run the worker loop.

        Pulls records from Redis, processes them with bounded concurrency,
        and publishes results back.

        Returns:
            PipelineResult with this worker's execution statistics.
        """
        aioredis = _import_redis()

        redis_queue = aioredis.from_url(self._redis_url, decode_responses=True)
        redis_pub = aioredis.from_url(self._redis_url, decode_responses=True)

        # Signal handling
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._request_shutdown)

        await self._checkpoint.load()

        sem = asyncio.Semaphore(self._concurrency)
        completed = 0
        rejected = 0
        failed = 0
        total = 0
        t0 = time.monotonic()
        tasks: list[asyncio.Task] = []

        async def _process_and_publish(raw: dict[str, Any]) -> None:
            nonlocal completed, rejected, failed
            async with sem:
                record_id = raw.get("id")
                record = (
                    DataRecord(seed_data=raw)
                    if record_id is None
                    else DataRecord(id=record_id, seed_data=raw)
                )

                # Skip if already done (race condition protection)
                if await self._checkpoint.is_done(record.id):
                    return

                record = await self._pipeline.process_record(record)

                # Publish result
                result_json = record.model_dump_json()
                await redis_pub.publish(self._result_channel, result_json)

                # Update checkpoint and counters
                if record.status == RecordStatus.COMPLETED:
                    await self._checkpoint.commit(record.id)
                    completed += 1
                elif record.status == RecordStatus.REJECTED:
                    rejected += 1
                elif record.status == RecordStatus.FAILED:
                    failed += 1

        try:
            async with self._checkpoint:
                while not self._shutdown:
                    # BRPOP with 1-second timeout to check shutdown flag
                    result = await redis_queue.brpop(self._queue_name, timeout=1)
                    if result is None:
                        # Timeout, check for shutdown or continue waiting
                        continue

                    _, raw_data = result
                    try:
                        raw = json.loads(raw_data)
                    except json.JSONDecodeError:
                        logger.warning("Malformed task data, skipping")
                        continue

                    # Check for poison pill
                    if raw.get("__poison__"):
                        logger.info("Worker %s received poison pill, shutting down", self._worker_id)
                        break

                    total += 1
                    task = asyncio.create_task(_process_and_publish(raw))
                    tasks.append(task)

                    # Clean up finished tasks periodically
                    if len(tasks) > self._concurrency * 2:
                        done = [t for t in tasks if t.done()]
                        for t in done:
                            tasks.remove(t)
                            # Re-raise any exceptions
                            if t.exception():
                                logger.error("Task failed: %s", t.exception())

                # Wait for remaining tasks to finish
                if tasks:
                    logger.info(
                        "Worker %s draining %d in-flight tasks...",
                        self._worker_id, len(tasks),
                    )
                    await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            await redis_queue.aclose()
            await redis_pub.aclose()

        elapsed = time.monotonic() - t0
        rps = total / elapsed if elapsed > 0 else 0.0

        logger.info(
            "Worker %s finished: %d processed (completed=%d, rejected=%d, failed=%d) "
            "in %.1fs (%.1f rec/s)",
            self._worker_id, total, completed, rejected, failed, elapsed, rps,
        )

        return PipelineResult(
            total_records=total,
            completed=completed,
            rejected=rejected,
            failed=failed,
            elapsed_seconds=elapsed,
            records_per_second=rps,
        )

    def _request_shutdown(self) -> None:
        if not self._shutdown:
            logger.info("Worker %s shutdown requested", self._worker_id)
            self._shutdown = True
