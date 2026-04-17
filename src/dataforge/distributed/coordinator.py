# src/dataforge/distributed/coordinator.py
"""Coordinator process for the distributed Coordinator-Worker mode.

The coordinator:
1. Reads input data and filters against the shared checkpoint
2. Pushes pending records to a Redis task queue (LPUSH)
3. Listens for results on a Redis Pub/Sub channel
4. Writes completed records to the output file
5. Tracks global progress and supports graceful shutdown

Usage::

    coordinator = Coordinator(config)
    result = await coordinator.run()
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
from dataforge.schema import RecordStatus

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


class Coordinator:
    """Distributed pipeline coordinator.

    Reads input, distributes records via Redis, collects results.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        run_id: str = "default",
        queue_name: str = "dataforge:tasks",
        result_channel: str = "dataforge:results",
    ) -> None:
        self._redis_url = redis_url
        self._run_id = run_id
        self._queue_name = queue_name
        self._result_channel = result_channel
        self._checkpoint = RedisCheckpointManager(redis_url, run_id=run_id)
        self._shutdown = False

    async def run(
        self,
        input_path: str,
        output_path: str,
        num_workers: int = 4,
    ) -> PipelineResult:
        """Run the coordinator loop.

        Args:
            input_path: Path to input JSONL file.
            output_path: Path to output JSONL file (appended).
            num_workers: Expected number of workers (for progress tracking).

        Returns:
            PipelineResult with execution statistics.
        """
        aioredis = _import_redis()

        redis_pub = aioredis.from_url(self._redis_url, decode_responses=True)
        redis_sub = aioredis.from_url(self._redis_url, decode_responses=True)

        # Setup signal handling
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self._request_shutdown)

        try:
            await self._checkpoint.load()
            t0 = time.monotonic()

            # Phase 1: Read input and enqueue pending records
            total_records = 0
            enqueued = 0
            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_records += 1
                    raw = json.loads(line)
                    record_id = raw.get("id")
                    if record_id and await self._checkpoint.is_done(record_id):
                        continue
                    # Serialize the raw dict for the queue
                    await redis_pub.lpush(self._queue_name, json.dumps(raw, ensure_ascii=False))
                    enqueued += 1

            already_done = total_records - enqueued
            logger.info(
                "Coordinator: %d total records, %d already done, %d enqueued",
                total_records, already_done, enqueued,
            )

            if enqueued == 0:
                elapsed = time.monotonic() - t0
                return PipelineResult(
                    total_records=total_records,
                    completed=already_done,
                    rejected=0,
                    failed=0,
                    elapsed_seconds=elapsed,
                    records_per_second=0.0,
                )

            # Phase 2: Listen for results via Pub/Sub
            pubsub = redis_sub.pubsub()
            await pubsub.subscribe(self._result_channel)

            completed = 0
            rejected = 0
            failed = 0
            processed = 0

            async with self._checkpoint:
                with open(output_path, "a", encoding="utf-8") as out_file:
                    while processed < enqueued and not self._shutdown:
                        msg = await pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=1.0
                        )
                        if msg is None:
                            continue
                        if msg["type"] != "message":
                            continue

                        try:
                            result_data: dict[str, Any] = json.loads(msg["data"])
                            record_id = result_data.get("id", "")
                            status = result_data.get("status", "")

                            if status == RecordStatus.COMPLETED.value:
                                out_file.write(json.dumps(result_data, ensure_ascii=False) + "\n")
                                await self._checkpoint.commit(record_id)
                                completed += 1
                            elif status == RecordStatus.REJECTED.value:
                                rejected += 1
                            elif status == RecordStatus.FAILED.value:
                                failed += 1

                            processed += 1

                            if processed % 100 == 0:
                                logger.info(
                                    "Progress: %d/%d (completed=%d, rejected=%d, failed=%d)",
                                    processed, enqueued, completed, rejected, failed,
                                )
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning("Malformed result message: %s", e)

            await pubsub.unsubscribe(self._result_channel)
            await pubsub.aclose()

            elapsed = time.monotonic() - t0
            total_completed = already_done + completed
            rps = enqueued / elapsed if elapsed > 0 else 0.0

            logger.info(
                "Coordinator finished: %d completed, %d rejected, %d failed in %.1fs (%.1f rec/s)",
                total_completed, rejected, failed, elapsed, rps,
            )

            return PipelineResult(
                total_records=total_records,
                completed=total_completed,
                rejected=rejected,
                failed=failed,
                elapsed_seconds=elapsed,
                records_per_second=rps,
            )

        finally:
            await redis_pub.aclose()
            await redis_sub.aclose()

    def _request_shutdown(self) -> None:
        if not self._shutdown:
            logger.info("Coordinator shutdown requested, waiting for in-flight results...")
            self._shutdown = True

    async def send_poison_pills(self, num_workers: int) -> None:
        """Send poison pill messages to gracefully stop workers."""
        aioredis = _import_redis()
        redis = aioredis.from_url(self._redis_url, decode_responses=True)
        try:
            for _ in range(num_workers):
                await redis.lpush(self._queue_name, json.dumps({"__poison__": True}))
            logger.info("Sent %d poison pills to workers", num_workers)
        finally:
            await redis.aclose()
