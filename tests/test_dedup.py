# tests/test_dedup.py
"""Tests for the semantic deduplication module."""

import pytest

from dataforge.dedup import (
    DedupEvaluator,
    SemanticDeduplicator,
    _cosine_similarity,
)
from dataforge.registry import get_evaluator
from dataforge.schema import DataRecord

# --- Helpers ---

class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [type("Obj", (), {"embedding": e})() for e in embeddings]


class MockEmbeddingClient:
    """Mock openai.AsyncOpenAI that returns fixed embeddings."""

    def __init__(self, embedding_map=None, default_embeddings=None):
        """
        Args:
            embedding_map: dict mapping text to embedding vector
            default_embeddings: list of embeddings returned in order
        """
        self.embeddings = type("NS", (), {"create": self._create})()
        self._embedding_map = embedding_map or {}
        self._default_embeddings = default_embeddings or []
        self._call_count = 0

    async def _create(self, model, input):
        results = []
        for text in input:
            if text in self._embedding_map:
                results.append(self._embedding_map[text])
            elif self._call_count < len(self._default_embeddings):
                results.append(self._default_embeddings[self._call_count])
                self._call_count += 1
            else:
                results.append([0.0, 0.0, 1.0])  # default
        return MockEmbeddingResponse(results)


# --- SemanticDeduplicator tests ---

async def test_dedup_removes_duplicates():
    records = [
        {"instruction": "hello world", "id": "1"},
        {"instruction": "hello world duplicate", "id": "2"},
        {"instruction": "something different", "id": "3"},
    ]
    dedup = SemanticDeduplicator(api_key="test-key", threshold=0.95)
    # Mock: first two texts have near-identical embeddings
    dedup._client = MockEmbeddingClient(embedding_map={
        "hello world": [1.0, 0.0, 0.0],
        "hello world duplicate": [0.99, 0.01, 0.0],  # very similar
        "something different": [0.0, 1.0, 0.0],  # orthogonal
    })

    result = await dedup.deduplicate(records)

    # "hello world duplicate" should be removed (sim ≈ 0.999)
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "3"


async def test_dedup_empty_list():
    dedup = SemanticDeduplicator(api_key="test-key")
    result = await dedup.deduplicate([])
    assert result == []


async def test_dedup_no_duplicates():
    records = [
        {"instruction": "topic A", "id": "1"},
        {"instruction": "topic B", "id": "2"},
    ]
    dedup = SemanticDeduplicator(api_key="test-key", threshold=0.95)
    dedup._client = MockEmbeddingClient(embedding_map={
        "topic A": [1.0, 0.0],
        "topic B": [0.0, 1.0],
    })

    result = await dedup.deduplicate(records)
    assert len(result) == 2


async def test_dedup_custom_field():
    records = [
        {"text": "same", "id": "1"},
        {"text": "same", "id": "2"},
    ]
    dedup = SemanticDeduplicator(api_key="test-key", field="text", threshold=0.99)
    dedup._client = MockEmbeddingClient(embedding_map={
        "same": [1.0, 0.0],
    })

    result = await dedup.deduplicate(records)
    # Both have identical embeddings, only first is kept
    assert len(result) == 1
    assert result[0]["id"] == "1"


# --- DedupEvaluator tests ---

async def test_dedup_evaluator_accepts_unique():
    evaluator = DedupEvaluator(api_key="test-key", threshold=0.95)
    evaluator._client = MockEmbeddingClient(embedding_map={
        "response A": [1.0, 0.0, 0.0],
    })

    record = DataRecord(
        seed_data={"instruction": "q1"},
        synthetic_data={"response": "response A"},
    )
    result = await evaluator.evaluate(record)
    assert result is True
    assert len(evaluator._seen_embeddings) == 1


async def test_dedup_evaluator_rejects_duplicate():
    evaluator = DedupEvaluator(api_key="test-key", threshold=0.95)
    evaluator._client = MockEmbeddingClient(embedding_map={
        "response A": [1.0, 0.0],
        "response A copy": [0.999, 0.001],
    })

    record1 = DataRecord(
        seed_data={"instruction": "q1"},
        synthetic_data={"response": "response A"},
    )
    record2 = DataRecord(
        seed_data={"instruction": "q2"},
        synthetic_data={"response": "response A copy"},
    )

    assert await evaluator.evaluate(record1) is True
    assert await evaluator.evaluate(record2) is False
    assert "dedup_similarity" in record2.metadata


async def test_dedup_evaluator_empty_text_rejected():
    evaluator = DedupEvaluator(api_key="test-key")
    evaluator._client = MockEmbeddingClient()

    record = DataRecord(
        seed_data={"instruction": "q"},
        synthetic_data={},
    )
    result = await evaluator.evaluate(record)
    assert result is False


async def test_dedup_evaluator_custom_field():
    evaluator = DedupEvaluator(api_key="test-key", field="answer", threshold=0.95)
    evaluator._client = MockEmbeddingClient(embedding_map={
        "unique answer": [0.0, 1.0, 0.0],
    })

    record = DataRecord(
        seed_data={"q": "test"},
        synthetic_data={"answer": "unique answer"},
    )
    result = await evaluator.evaluate(record)
    assert result is True


def test_dedup_registry_registration():
    cls = get_evaluator("dedup")
    assert cls is DedupEvaluator


def test_cosine_similarity_function():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
