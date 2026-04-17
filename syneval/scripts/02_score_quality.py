#!/usr/bin/env python3
"""SynEval Step 2: Score synthesized data on 5 quality dimensions.

Dimensions:
  D1 completeness  - instruction + response both non-empty (rule-based)
  D2 length        - token count within [min_tokens, max_tokens] (rule-based)
  D3 llm_score     - LLM-as-Judge multi-criteria score 1-5
  D4 similarity    - cosine similarity to seed within [sim_min, sim_max]
  D5 dedup         - exact + fuzzy 5-gram deduplication

Each record in the output JSONL gets a ``quality_scores`` dict appended to
its ``metadata`` field. A ``sft_readiness_score`` (0-100) is also computed.

Usage::

    python syneval/scripts/02_score_quality.py \
        --dataset syneval/data/synthesized_50k.jsonl \
        --output-dir syneval/data \
        --vllm-url http://localhost:8100/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --openai-api-key $OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D1: Completeness
# ---------------------------------------------------------------------------
def score_completeness(record: dict[str, Any]) -> dict[str, Any]:
    """Rule-based check that instruction and response are non-empty."""
    synth = record.get("synthetic_data") or record
    instruction = str(synth.get("instruction", "")).strip()
    response = str(synth.get("response", synth.get("output", ""))).strip()
    passed = bool(instruction) and bool(response)
    return {"passed": passed, "score": 1.0 if passed else 0.0}


# ---------------------------------------------------------------------------
# D2: Length filter
# ---------------------------------------------------------------------------
def score_length(
    record: dict[str, Any],
    min_tokens: int = 10,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Rough token count (whitespace split) within bounds."""
    synth = record.get("synthetic_data") or record
    response = str(synth.get("response", synth.get("output", ""))).strip()
    token_count = len(response.split())
    passed = min_tokens <= token_count <= max_tokens
    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "token_count": token_count,
    }


# ---------------------------------------------------------------------------
# D3: LLM-as-Judge (multi-criteria)
# ---------------------------------------------------------------------------
async def score_llm_judge_batch(
    records: list[dict[str, Any]],
    llm_client: Any,
    batch_size: int = 50,
) -> list[dict[str, Any]]:
    """Score records with LLM-as-Judge in batches."""
    from dataforge.evaluators.multi_criteria import MultiCriteriaEvaluator
    from dataforge.schema import DataRecord, RecordStatus

    evaluator = MultiCriteriaEvaluator(llm=llm_client)
    results: list[dict[str, Any]] = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        tasks = []
        for k, rec in enumerate(batch):
            synth = rec.get("synthetic_data") or rec
            dr = DataRecord(
                id=rec.get("id", f"rec_{i + k}"),
                seed_data={
                    "instruction": str(synth.get("instruction", "")),
                },
                synthetic_data={
                    "instruction": str(synth.get("instruction", "")),
                    "response": str(synth.get("response", synth.get("output", ""))),
                },
                status=RecordStatus.GENERATED,
            )
            tasks.append(evaluator.assess(dr))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.warning("LLM judge error for record %d: %s", i + j, result)
                results.append({"passed": False, "score": None, "error": str(result)})
            else:
                results.append({
                    "passed": result.passed,
                    "score": result.score,
                    "details": result.details,
                })

        if (i // batch_size) % 10 == 0:
            logger.info(
                "LLM judge progress: %d/%d records scored", min(i + batch_size, len(records)), len(records)
            )

    return results


# ---------------------------------------------------------------------------
# D4: Semantic similarity
# ---------------------------------------------------------------------------
async def score_similarity_batch(
    records: list[dict[str, Any]],
    embedding_client: Any,
    min_sim: float = 0.3,
    max_sim: float = 0.95,
    batch_size: int = 100,
    embedding_model: str = "text-embedding-v4",
) -> list[dict[str, Any]]:
    """Score semantic similarity between seed instruction and synthetic response."""
    from dataforge.evaluators.similarity import SimilarityEvaluator
    from dataforge.schema import DataRecord, RecordStatus

    evaluator = SimilarityEvaluator(
        api_key="dummy",
        embedding_model=embedding_model,
        min_similarity=min_sim,
        max_similarity=max_sim,
    )
    # Override with the properly configured client
    evaluator._client = embedding_client

    results: list[dict[str, Any]] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        tasks = []
        for k, rec in enumerate(batch):
            synth = rec.get("synthetic_data") or rec
            seed = rec.get("seed_data") or {}
            dr = DataRecord(
                id=rec.get("id", f"rec_{i + k}"),
                seed_data={"instruction": str(seed.get("instruction", synth.get("instruction", "")))},
                synthetic_data={
                    "instruction": str(synth.get("instruction", "")),
                    "response": str(synth.get("response", synth.get("output", ""))),
                },
                status=RecordStatus.GENERATED,
            )
            tasks.append(evaluator.assess(dr))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.warning("Similarity error for record %d: %s", i + j, result)
                results.append({"passed": False, "score": None, "error": str(result)})
            else:
                results.append({
                    "passed": result.passed,
                    "score": result.score,
                    "details": result.details,
                })

        if (i // batch_size) % 5 == 0:
            logger.info(
                "Similarity progress: %d/%d records scored", min(i + batch_size, len(records)), len(records)
            )

    return results


# ---------------------------------------------------------------------------
# D5: Deduplication
# ---------------------------------------------------------------------------
def build_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = text.lower().split()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def score_deduplication(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark duplicates using exact hash and fuzzy 5-gram Jaccard similarity."""
    seen_hashes: set[str] = set()
    seen_ngrams: list[tuple[set, int]] = []  # (ngram_set, original_index)
    results: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        synth = rec.get("synthetic_data") or rec
        response = str(synth.get("response", synth.get("output", ""))).strip()

        # Exact dedup
        import hashlib
        exact_hash = hashlib.md5(response.encode()).hexdigest()
        if exact_hash in seen_hashes:
            results.append({"passed": False, "score": 0.0, "reason": "exact_duplicate"})
            continue

        # Fuzzy dedup (Jaccard >= 0.8 = near-duplicate)
        ngrams = build_ngrams(response)
        is_near_dup = False
        for prev_ngrams, _ in seen_ngrams:
            if not prev_ngrams or not ngrams:
                continue
            intersection = len(ngrams & prev_ngrams)
            union = len(ngrams | prev_ngrams)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard >= 0.8:
                is_near_dup = True
                break

        if is_near_dup:
            results.append({"passed": False, "score": 0.0, "reason": "near_duplicate"})
        else:
            seen_hashes.add(exact_hash)
            seen_ngrams.append((ngrams, i))
            results.append({"passed": True, "score": 1.0, "reason": None})

    dup_count = sum(1 for r in results if not r["passed"])
    logger.info("Deduplication: %d/%d duplicates found (%.1f%%)",
                dup_count, len(records), 100.0 * dup_count / max(len(records), 1))
    return results


# ---------------------------------------------------------------------------
# SFT Readiness Score (weighted aggregate)
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS = {
    "completeness": 0.10,
    "length": 0.10,
    "llm_score": 0.50,
    "similarity": 0.15,
    "dedup": 0.15,
}


def compute_sft_readiness(scores: dict[str, Any]) -> float:
    """Compute weighted SFT readiness score 0-100."""
    total = 0.0
    weight_sum = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        dim_result = scores.get(dim, {})
        raw_score = dim_result.get("score")
        if raw_score is not None:
            # Normalize: completeness/length/dedup are 0-1, llm_score is 1-5, similarity is 0-1
            if dim == "llm_score":
                normalized = (float(raw_score) - 1.0) / 4.0  # map 1-5 → 0-1
            else:
                normalized = float(raw_score)
            total += weight * normalized * 100
            weight_sum += weight
    return round(total / weight_sum, 2) if weight_sum > 0 else 0.0


# ---------------------------------------------------------------------------
# Main scoring pipeline
# ---------------------------------------------------------------------------
async def score_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    logger.info("Loading records from %s", args.dataset)
    records: list[dict[str, Any]] = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records", len(records))

    # D1: Completeness (sync)
    logger.info("Scoring D1: Completeness...")
    d1_results = [score_completeness(r) for r in records]

    # D2: Length (sync)
    logger.info("Scoring D2: Length...")
    d2_results = [score_length(r, args.min_tokens, args.max_tokens) for r in records]

    # D5: Deduplication (sync, must run before LLM calls to be efficient)
    logger.info("Scoring D5: Deduplication...")
    d5_results = score_deduplication(records)

    # D3: LLM Judge (async)
    d3_results: list[dict[str, Any]] = [{"passed": False, "score": None}] * len(records)
    if not args.skip_llm_judge:
        logger.info("Scoring D3: LLM Judge (this may take a while)...")
        from dataforge.clients.vllm_client import vLLMClient
        llm = vLLMClient(
            model=args.model,
            base_url=args.vllm_url,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
        )
        d3_results = await score_llm_judge_batch(records, llm, batch_size=args.batch_size)

    # D4: Similarity (async)
    d4_results: list[dict[str, Any]] = [{"passed": False, "score": None}] * len(records)
    if not args.skip_similarity:
        logger.info("Scoring D4: Similarity (embedding API: %s)...", args.openai_base_url)
        import openai
        emb_client = openai.AsyncOpenAI(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
        )
        d4_results = await score_similarity_batch(
            records, emb_client,
            min_sim=args.sim_min,
            max_sim=args.sim_max,
            batch_size=args.batch_size,
            embedding_model=args.embedding_model,
        )

    # Combine all scores and write output
    logger.info("Writing scored dataset to %s", output_dir)
    output_path = output_dir / "scored_50k.jsonl"
    score_summary: dict[str, Any] = collections.defaultdict(lambda: {"total": 0, "passed": 0, "scores": []})

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            quality_scores = {
                "completeness": d1_results[i],
                "length": d2_results[i],
                "llm_score": d3_results[i],
                "similarity": d4_results[i],
                "dedup": d5_results[i],
            }
            sft_readiness = compute_sft_readiness(quality_scores)

            # Add scores to the record metadata
            rec_out = dict(rec)
            meta = rec_out.setdefault("metadata", {})
            meta["quality_scores"] = quality_scores
            meta["sft_readiness_score"] = sft_readiness

            out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

            # Accumulate summary stats
            for dim, result in quality_scores.items():
                score_summary[dim]["total"] += 1
                if result.get("passed"):
                    score_summary[dim]["passed"] += 1
                if result.get("score") is not None:
                    score_summary[dim]["scores"].append(result["score"])

    # Compute and save summary
    summary: dict[str, Any] = {
        "total_records": len(records),
        "output_path": str(output_path),
        "dimensions": {},
    }
    for dim, stats in score_summary.items():
        total = stats["total"]
        passed = stats["passed"]
        scores_list = stats["scores"]
        summary["dimensions"][dim] = {
            "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
            "avg_score": round(sum(scores_list) / len(scores_list), 4) if scores_list else None,
            "passed": passed,
            "total": total,
        }

    summary_path = output_dir / "quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Quality summary written to %s", summary_path)
    logger.info("Scoring complete.")
    for dim, stats in summary["dimensions"].items():
        logger.info(
            "  D%-12s pass_rate=%.1f%%  avg_score=%s",
            dim,
            stats["pass_rate"] * 100,
            f"{stats['avg_score']:.3f}" if stats["avg_score"] is not None else "N/A",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SynEval Step 2: Score synthesized data on 5 quality dimensions."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to synthesized JSONL dataset.")
    parser.add_argument("--output-dir", type=str, default="syneval/data",
                        help="Directory for scored output.")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8100/v1",
                        help="vLLM server URL for LLM judge.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for LLM judge.")
    parser.add_argument("--openai-api-key", type=str, default=None,
                        help="API key for embeddings (D4 similarity). Works with any OpenAI-compatible API.")
    parser.add_argument("--openai-base-url", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="Base URL for embeddings API. Default: DashScope compatible endpoint.")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-v4",
                        help="Embedding model name. Default: text-embedding-v4 (DashScope).")
    parser.add_argument("--min-tokens", type=int, default=10,
                        help="Minimum response token count for length filter.")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum response token count for length filter.")
    parser.add_argument("--sim-min", type=float, default=0.3,
                        help="Minimum cosine similarity threshold.")
    parser.add_argument("--sim-max", type=float, default=0.95,
                        help="Maximum cosine similarity threshold.")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Async batch size for LLM/embedding calls.")
    parser.add_argument("--rpm-limit", type=int, default=1000)
    parser.add_argument("--tpm-limit", type=int, default=1_000_000)
    parser.add_argument("--skip-llm-judge", action="store_true",
                        help="Skip D3 LLM judge scoring (for fast testing).")
    parser.add_argument("--skip-similarity", action="store_true",
                        help="Skip D4 similarity scoring (no embedding API needed).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(score_all(args))


if __name__ == "__main__":
    main()
