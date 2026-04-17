#!/usr/bin/env python3
"""SynEval Step 1: Synthesize 50K training records from Alpaca-52K seeds.

Uses DataForge EvolInstruct pipeline with Qwen2.5-7B-Instruct via vLLM.
Aligns with AlpaGasus 52K scale for direct comparison.

Usage::

    python syneval/scripts/01_synthesize.py \
        --seed-dataset experiments/seeds/seeds_10k.jsonl \
        --target 50000 \
        --vllm-url http://localhost:8100/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-dir syneval/data \
        --concurrency 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from dataforge.clients.vllm_client import vLLMClient
from dataforge.evaluators.regex_filter import RegexFilter
from dataforge.pipeline import Pipeline
from dataforge.strategies.evol_instruct import EvolInstruct

logger = logging.getLogger(__name__)

# Round-robin across multiple vLLM ports for maximum throughput
DEFAULT_VLLM_PORTS = list(range(8100, 8108))


def make_urls(base_host: str, ports: list[int]) -> list[str]:
    return [f"http://{base_host}:{p}/v1" for p in ports]


async def synthesize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "synthesized_50k.jsonl")

    # Build round-robin URL list
    if args.vllm_url:
        urls = [args.vllm_url]
    else:
        urls = make_urls(args.vllm_host, args.vllm_ports)

    logger.info("Using vLLM endpoints: %s", urls)

    # Use first URL; for multi-URL, clients are created per-port
    base_url = urls[0]
    llm = vLLMClient(
        model=args.model,
        base_url=base_url,
        rpm_limit=args.rpm_limit,
        tpm_limit=args.tpm_limit,
    )

    strategy = EvolInstruct(llm=llm, depth=args.evol_depth)

    # Minimal regex filter to remove obviously bad outputs
    regex_filter = RegexFilter(
        blacklist_patterns=[
            r"(?i)as an ai language model",
            r"(?i)i cannot (and will not|help with)",
            r"(?i)i'm sorry, (but )?i (can't|cannot|am unable)",
        ]
    )

    checkpoint_dir = tempfile.mkdtemp(prefix="syneval_synthesis_")
    pipeline = Pipeline(
        strategy=strategy,
        evaluators=[regex_filter],
        checkpoint_dir=checkpoint_dir,
        max_retries=3,
    )

    logger.info(
        "Starting synthesis: seed=%s target=%d output=%s",
        args.seed_dataset,
        args.target,
        output_path,
    )

    result = await pipeline.run(
        input_path=args.seed_dataset,
        output_path=output_path,
        concurrency=args.concurrency,
        show_progress=True,
    )

    logger.info(
        "Synthesis complete: completed=%d rejected=%d failed=%d elapsed=%.1fs",
        result.completed,
        result.rejected,
        result.failed,
        result.elapsed_seconds,
    )

    # Write synthesis metadata
    meta = {
        "seed_dataset": args.seed_dataset,
        "target": args.target,
        "completed": result.completed,
        "rejected": result.rejected,
        "failed": result.failed,
        "elapsed_seconds": result.elapsed_seconds,
        "records_per_second": result.records_per_second,
        "output_path": output_path,
        "model": args.model,
        "evol_depth": args.evol_depth,
    }
    meta_path = output_dir / "synthesis_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Synthesis metadata written to %s", meta_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SynEval Step 1: Synthesize 50K training records."
    )
    parser.add_argument(
        "--seed-dataset",
        type=str,
        required=True,
        help="Path to seeds JSONL (e.g. experiments/seeds/seeds_10k.jsonl).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50000,
        help="Target number of synthesized records (default: 50000).",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="Single vLLM server URL. Overrides --vllm-host/--vllm-ports.",
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default="localhost",
        help="vLLM server hostname (default: localhost).",
    )
    parser.add_argument(
        "--vllm-ports",
        type=int,
        nargs="+",
        default=DEFAULT_VLLM_PORTS,
        help="vLLM server ports for round-robin (default: 8100-8107).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name served by vLLM.",
    )
    parser.add_argument(
        "--evol-depth",
        type=int,
        default=2,
        help="EvolInstruct mutation depth (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="syneval/data",
        help="Directory to write synthesized dataset.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent pipeline workers (default: 50).",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=2000,
        help="Requests per minute limit (default: 2000).",
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=2_000_000,
        help="Tokens per minute limit (default: 2M).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(synthesize(args))


if __name__ == "__main__":
    main()
