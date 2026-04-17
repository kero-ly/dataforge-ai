# tests/test_similarity.py
"""Tests for the SimilarityEvaluator."""
import math

import pytest

from dataforge.evaluators.similarity import SimilarityEvaluator, _cosine_similarity
from dataforge.registry import get_evaluator
from dataforge.schema import DataRecord

# --- Cosine similarity unit tests ---

def test_cosine_identical_vectors():
    assert _cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_arbitrary_vectors():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
    norm_a = math.sqrt(1 + 4 + 9)  # sqrt(14)
    norm_b = math.sqrt(16 + 25 + 36)  # sqrt(77)
    expected = dot / (norm_a * norm_b)
    assert _cosine_similarity(a, b) == pytest.approx(expected)


# --- Registry ---

def test_registry_registration():
    cls = get_evaluator("similarity")
    assert cls is SimilarityEvaluator


# --- Evaluator integration (mocked embeddings) ---

class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [type("Obj", (), {"embedding": e})() for e in embeddings]


class MockEmbeddingClient:
    """Mock openai.AsyncOpenAI that returns fixed embeddings."""
    def __init__(self, embeddings):
        self.embeddings = type("NS", (), {"create": self._create})()
        self._fixed_embeddings = embeddings

    async def _create(self, model, input):
        return MockEmbeddingResponse(self._fixed_embeddings[:len(input)])


async def test_evaluator_accepts_similar():
    evaluator = SimilarityEvaluator(api_key="test-key", min_similarity=0.5, max_similarity=1.0)
    # Mock the client to return near-identical embeddings
    evaluator._client = MockEmbeddingClient([[1.0, 0.0], [0.9, 0.1]])

    record = DataRecord(
        seed_data={"instruction": "hello"},
        synthetic_data={"instruction": "hi there"},
    )
    result = await evaluator.evaluate(record)
    assert result is True
    assert "similarity_score" in record.metadata


async def test_evaluator_rejects_dissimilar():
    evaluator = SimilarityEvaluator(api_key="test-key", min_similarity=0.8, max_similarity=1.0)
    # Orthogonal vectors → cosine = 0
    evaluator._client = MockEmbeddingClient([[1.0, 0.0], [0.0, 1.0]])

    record = DataRecord(
        seed_data={"instruction": "hello"},
        synthetic_data={"instruction": "completely unrelated"},
    )
    result = await evaluator.evaluate(record)
    assert result is False


async def test_evaluator_rejects_too_similar():
    evaluator = SimilarityEvaluator(api_key="test-key", min_similarity=0.3, max_similarity=0.8)
    # Identical vectors → cosine = 1.0, exceeds max
    evaluator._client = MockEmbeddingClient([[1.0, 0.0], [1.0, 0.0]])

    record = DataRecord(
        seed_data={"instruction": "exact copy"},
        synthetic_data={"instruction": "exact copy"},
    )
    result = await evaluator.evaluate(record)
    assert result is False


async def test_evaluator_empty_text_rejects():
    evaluator = SimilarityEvaluator(api_key="test-key")
    evaluator._client = MockEmbeddingClient([[1.0], [1.0]])

    record = DataRecord(
        seed_data={"instruction": ""},
        synthetic_data={"response": "some data"},
    )
    result = await evaluator.evaluate(record)
    assert result is False


async def test_evaluator_custom_seed_field():
    evaluator = SimilarityEvaluator(api_key="test-key", seed_field="text", min_similarity=0.0, max_similarity=1.0)
    evaluator._client = MockEmbeddingClient([[1.0, 0.5], [0.8, 0.6]])

    record = DataRecord(
        seed_data={"text": "source passage"},
        synthetic_data={"response": "generated output"},
    )
    result = await evaluator.evaluate(record)
    assert result is True
