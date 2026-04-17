# src/dataforge/engine/redis_checkpoint.py
"""Redis-backed checkpoint manager for distributed pipelines.

Drop-in replacement for CheckpointManager that uses a Redis SET
to track completed record IDs across multiple worker processes.
"""
from __future__ import annotations

import logging
from types import TracebackType

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


class RedisCheckpointManager:
    """Distributed checkpoint manager backed by Redis SET.

    Uses ``SADD``/``SISMEMBER`` for O(1) commit/lookup. The Redis SET
    key is ``dataforge:checkpoint:{run_id}``.

    Implements the same interface as CheckpointManager::

        mgr = RedisCheckpointManager(redis_url, run_id="exp_001")
        await mgr.load()
        async with mgr:
            await mgr.commit(record_id)
            assert await mgr.is_done(record_id)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        run_id: str = "default",
    ) -> None:
        self._redis_url = redis_url
        self._run_id = run_id
        self._key = f"dataforge:checkpoint:{run_id}"
        self._redis = None
        self._count: int = 0

    async def __aenter__(self) -> RedisCheckpointManager:
        aioredis = _import_redis()
        self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def load(self) -> None:
        """Load the count of completed IDs from Redis."""
        aioredis = _import_redis()
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        self._count = await self._redis.scard(self._key)
        if self._count > 0:
            logger.info(
                "Redis checkpoint %s: %d records already completed",
                self._key, self._count,
            )

    async def commit(self, record_id: str) -> None:
        """Mark a record ID as completed."""
        assert self._redis is not None, "Call load() or use async with first"
        added = await self._redis.sadd(self._key, record_id)
        if added:
            self._count += 1

    async def commit_batch(self, record_ids: list[str]) -> None:
        """Mark multiple record IDs as completed in a single SADD."""
        if not record_ids:
            return
        assert self._redis is not None, "Call load() or use async with first"
        added = await self._redis.sadd(self._key, *record_ids)
        self._count += added

    async def is_done(self, record_id: str) -> bool:
        """Check if a record ID has been completed."""
        assert self._redis is not None, "Call load() or use async with first"
        return bool(await self._redis.sismember(self._key, record_id))

    @property
    def completed_count(self) -> int:
        """Number of completed record IDs."""
        return self._count

    async def clear(self) -> None:
        """Remove all completed IDs (for testing)."""
        if self._redis is not None:
            await self._redis.delete(self._key)
            self._count = 0
