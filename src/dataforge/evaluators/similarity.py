# src/dataforge/evaluators/similarity.py
from __future__ import annotations

import logging
import math
import time

import openai

from dataforge.assessment.schema import AssessmentResult
from dataforge.evaluators.base import BaseEvaluator
from dataforge.registry import register_evaluator
from dataforge.schema import DataRecord

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same dimensionality")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@register_evaluator("similarity")
class SimilarityEvaluator(BaseEvaluator):
    """Semantic similarity evaluator using embedding cosine similarity.

    Compares seed data text against synthetic data text using embeddings.
    Rejects records where similarity falls outside the [min, max] range.

    Args:
        api_key: OpenAI API key for embeddings. Falls back to OPENAI_API_KEY env var.
        base_url: Optional base URL for the embedding API.
        embedding_model: Embedding model name. Default "text-embedding-3-small".
        min_similarity: Minimum cosine similarity threshold. Default 0.3.
        max_similarity: Maximum cosine similarity threshold. Default 0.95.
        seed_field: Field in seed_data to embed. Default "instruction".
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        min_similarity: float = 0.3,
        max_similarity: float = 0.95,
        seed_field: str = "instruction",
    ) -> None:
        self.embedding_model = embedding_model
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.seed_field = seed_field
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Fetch embeddings for a batch of texts."""
        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        started = time.monotonic()
        seed_text = str(record.seed_data.get(self.seed_field, ""))
        synthetic_text = self._get_content(record)

        if not seed_text or not synthetic_text:
            logger.debug("Empty text for record %s, rejecting", record.id)
            return AssessmentResult(
                evaluator=type(self).__name__,
                passed=False,
                threshold=self.min_similarity,
                reason_codes=["empty_text"],
                details={"seed_field": self.seed_field},
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )

        embeddings = await self._get_embeddings([seed_text, synthetic_text])
        similarity = _cosine_similarity(embeddings[0], embeddings[1])

        record.metadata["similarity_score"] = round(similarity, 4)
        logger.debug(
            "Record %s similarity=%.4f (range [%.2f, %.2f])",
            record.id,
            similarity,
            self.min_similarity,
            self.max_similarity,
        )

        passed = self.min_similarity <= similarity <= self.max_similarity
        reason_codes: list[str] = []
        if similarity < self.min_similarity:
            reason_codes.append("below_min_similarity")
        if similarity > self.max_similarity:
            reason_codes.append("above_max_similarity")
        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=passed,
            score=round(similarity, 4),
            threshold=self.min_similarity,
            reason_codes=reason_codes,
            details={
                "min_similarity": self.min_similarity,
                "max_similarity": self.max_similarity,
                "seed_field": self.seed_field,
            },
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )
