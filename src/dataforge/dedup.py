# src/dataforge/dedup.py
"""Semantic deduplication using embedding-based similarity.

Provides both a standalone :class:`SemanticDeduplicator` for batch
post-processing and an inline :class:`DedupEvaluator` that rejects records
too similar to previously accepted ones during pipeline execution.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

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


class SemanticDeduplicator:
    """Batch deduplication of records based on embedding similarity.

    Computes embeddings for a designated text field, then removes records
    whose embedding is too close to an already-accepted record.

    Usage::

        dedup = SemanticDeduplicator(api_key="<your OPENAI_API_KEY>", threshold=0.95)
        unique_records = await dedup.deduplicate(records)

    Args:
        api_key: OpenAI API key for the embedding model.
        base_url: Optional base URL for the embedding API.
        embedding_model: Embedding model name.
        threshold: Cosine similarity above which a record is considered duplicate.
        field: Record dict key whose value is used for dedup comparison.
        batch_size: How many texts to embed per API call (max 2048).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        threshold: float = 0.95,
        field: str = "instruction",
        batch_size: int = 100,
    ) -> None:
        self.threshold = threshold
        self.field = field
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _get_embeddings_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Fetch embeddings for a batch of texts."""
        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def _get_all_embeddings(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Fetch embeddings for all texts, batching as needed."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = await self._get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def deduplicate(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove semantically duplicate records.

        Args:
            records: List of record dicts (e.g. loaded from JSONL).

        Returns:
            Deduplicated list preserving order of first occurrence.
        """
        if not records:
            return []

        texts = [str(r.get(self.field, "")) for r in records]
        embeddings = await self._get_all_embeddings(texts)

        # Greedy dedup: keep a record if it's not too similar to any kept record
        kept_indices: list[int] = []
        kept_embeddings: list[list[float]] = []

        for i, emb in enumerate(embeddings):
            is_dup = False
            for kept_emb in kept_embeddings:
                sim = _cosine_similarity(emb, kept_emb)
                if sim >= self.threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept_indices.append(i)
                kept_embeddings.append(emb)

        removed = len(records) - len(kept_indices)
        if removed > 0:
            logger.info(
                "Dedup: removed %d duplicates from %d records (threshold=%.3f)",
                removed,
                len(records),
                self.threshold,
            )

        return [records[i] for i in kept_indices]


@register_evaluator("dedup")
class DedupEvaluator(BaseEvaluator):
    """Inline dedup evaluator for use within a pipeline.

    Maintains a running index of accepted record embeddings and rejects
    any record whose text is too similar to a previously accepted one.

    .. note::

        Because the evaluator maintains state across records, it is
        **not safe** to share a single instance across multiple pipelines.

    Args:
        api_key: OpenAI API key for the embedding model.
        base_url: Optional base URL for the embedding API.
        embedding_model: Embedding model name.
        threshold: Cosine similarity above which a record is rejected.
        field: Synthetic data field to embed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        threshold: float = 0.95,
        field: str | None = None,
    ) -> None:
        self.threshold = threshold
        self.field = field
        self.embedding_model = embedding_model
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._seen_embeddings: list[list[float]] = []
        self._lock = asyncio.Lock()

    async def evaluate(self, record: DataRecord) -> bool:
        result = await self.assess(record)
        return result.passed

    async def assess(self, record: DataRecord) -> AssessmentResult:
        started = time.monotonic()
        if self.field:
            text = str((record.synthetic_data or {}).get(self.field, ""))
        else:
            text = self._get_content(record)

        if not text:
            return AssessmentResult(
                evaluator=type(self).__name__,
                passed=False,
                threshold=self.threshold,
                reason_codes=["empty_text"],
                duration_ms=round((time.monotonic() - started) * 1000.0, 3),
            )

        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=[text],
        )
        embedding = response.data[0].embedding

        async with self._lock:
            for seen_emb in self._seen_embeddings:
                sim = _cosine_similarity(embedding, seen_emb)
                if sim >= self.threshold:
                    record.metadata["dedup_similarity"] = round(sim, 4)
                    logger.debug(
                        "Record %s rejected as duplicate (sim=%.4f >= %.4f)",
                        record.id,
                        sim,
                        self.threshold,
                    )
                    return AssessmentResult(
                        evaluator=type(self).__name__,
                        passed=False,
                        score=round(sim, 4),
                        threshold=self.threshold,
                        reason_codes=["near_duplicate"],
                        details={"field": self.field},
                        duration_ms=round((time.monotonic() - started) * 1000.0, 3),
                    )

            self._seen_embeddings.append(embedding)

        return AssessmentResult(
            evaluator=type(self).__name__,
            passed=True,
            score=1.0,
            threshold=self.threshold,
            details={"field": self.field},
            duration_ms=round((time.monotonic() - started) * 1000.0, 3),
        )
