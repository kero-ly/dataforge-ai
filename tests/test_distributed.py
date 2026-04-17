# tests/test_distributed.py
"""Tests for the distributed coordinator-worker mode.

Tests that don't require Redis test the components in isolation.
Tests requiring Redis are marked with @pytest.mark.skipif.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataforge.config.schema import DistributedConfig, ForgeConfig
from dataforge.pipeline import Pipeline
from dataforge.schema import DataRecord, RecordStatus
from dataforge.strategies.base import BaseStrategy


class FakeStrategy(BaseStrategy):
    """Strategy that echoes seed_data as synthetic_data."""

    async def apply(self, record: DataRecord) -> DataRecord:
        record.synthetic_data = {"response": f"Echo: {record.seed_data.get('instruction', '')}"}
        return record


class TestProcessRecord:
    """Test the extracted process_record method."""

    async def test_process_record_completed(self) -> None:
        pipeline = Pipeline(strategy=FakeStrategy(), evaluators=[])
        record = DataRecord(seed_data={"instruction": "Hello"})
        result = await pipeline.process_record(record)
        assert result.status == RecordStatus.COMPLETED
        assert result.synthetic_data is not None

    async def test_process_record_rejected(self) -> None:
        from dataforge.evaluators.base import BaseEvaluator

        class RejectAll(BaseEvaluator):
            async def evaluate(self, record: DataRecord) -> bool:
                return False

        pipeline = Pipeline(strategy=FakeStrategy(), evaluators=[RejectAll()])
        record = DataRecord(seed_data={"instruction": "Hello"})
        result = await pipeline.process_record(record)
        assert result.status == RecordStatus.REJECTED

    async def test_process_record_failed(self) -> None:
        class FailingStrategy(BaseStrategy):
            async def apply(self, record: DataRecord) -> DataRecord:
                raise RuntimeError("LLM error")

        pipeline = Pipeline(strategy=FailingStrategy(), evaluators=[], max_retries=1)
        record = DataRecord(seed_data={"instruction": "Hello"})
        result = await pipeline.process_record(record)
        assert result.status == RecordStatus.FAILED
        assert "LLM error" in result.metadata.get("error", "")


class TestDistributedConfig:
    """Test config schema for distributed mode."""

    def test_default_config(self) -> None:
        config = DistributedConfig()
        assert config.enabled is False
        assert config.backend == "redis"
        assert config.redis_url == "redis://localhost:6379"

    def test_forge_config_without_distributed(self) -> None:
        config = ForgeConfig(
            name="test",
            source={"type": "jsonl", "path": "test.jsonl"},
            pipeline=[{
                "step": "generate",
                "strategy": "evol-instruct",
                "llm": {"provider": "vllm", "model": "test"},
            }],
            sink={"path": "output.jsonl"},
        )
        assert config.distributed is None

    def test_forge_config_with_distributed(self) -> None:
        config = ForgeConfig(
            name="test",
            source={"type": "jsonl", "path": "test.jsonl"},
            pipeline=[{
                "step": "generate",
                "strategy": "evol-instruct",
                "llm": {"provider": "vllm", "model": "test"},
            }],
            sink={"path": "output.jsonl"},
            distributed={
                "enabled": True,
                "redis_url": "redis://myhost:6379",
                "role": "coordinator",
            },
        )
        assert config.distributed is not None
        assert config.distributed.enabled is True
        assert config.distributed.redis_url == "redis://myhost:6379"
        assert config.distributed.role == "coordinator"


class TestRedisCheckpointImport:
    """Test that Redis checkpoint can be imported without redis installed."""

    def test_import_error_message(self) -> None:
        # This test verifies the import error message is helpful
        from dataforge.engine.redis_checkpoint import _import_redis
        # If redis is installed, this will succeed; if not, it should raise ImportError
        try:
            _import_redis()
        except ImportError as e:
            assert "pip install" in str(e)


class TestRedisRateLimiterImport:
    """Test that Redis rate limiter can be imported without redis installed."""

    def test_import_error_message(self) -> None:
        from dataforge.engine.redis_rate_limiter import _import_redis
        try:
            _import_redis()
        except ImportError as e:
            assert "pip install" in str(e)
