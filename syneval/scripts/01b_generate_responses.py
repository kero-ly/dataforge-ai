#!/usr/bin/env python3
"""SynEval Step 1b: Generate responses for evolved instructions.

After EvolInstruct produces evolved instructions (01_synthesize.py), this script
asks the LLM to generate a response for each instruction, producing the complete
instruction-response pairs needed for SFT.

Usage::

    python syneval/scripts/01b_generate_responses.py \
        --dataset syneval/data/synthesized_50k.jsonl \
        --output-dir syneval/data \
        --vllm-url http://localhost:8100/v1 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --concurrency 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from dataforge.clients.vllm_client import vLLMClient

logger = logging.getLogger(__name__)

_RESPONSE_PROMPT = """\
Please provide a helpful, accurate, and detailed response to the following instruction.

Instruction: {instruction}

Response:"""


async def generate_response(
    llm: vLLMClient,
    instruction: str,
    semaphore: asyncio.Semaphore,
) -> str:
    prompt = _RESPONSE_PROMPT.format(instruction=instruction)
    async with semaphore:
        return await llm.generate(prompt)


async def generate_all(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.dataset)
    output_path = output_dir / input_path.name  # overwrite in-place (or new name)
    if args.output_file:
        output_path = output_dir / args.output_file

    # Load existing records
    logger.info("Loading records from %s", input_path)
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records", len(records))

    # Find records that already have a response (resume support)
    needs_response = []
    for i, rec in enumerate(records):
        synth = rec.get("synthetic_data") or rec
        response = str(synth.get("response", synth.get("output", ""))).strip()
        if not response:
            needs_response.append(i)

    logger.info("%d records need response generation", len(needs_response))
    if not needs_response:
        logger.info("All records already have responses. Nothing to do.")
        return

    llm = vLLMClient(
        model=args.model,
        base_url=args.vllm_url,
        rpm_limit=args.rpm_limit,
        tpm_limit=args.tpm_limit,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    started = time.monotonic()
    batch_size = args.batch_size
    done = 0

    for batch_start in range(0, len(needs_response), batch_size):
        batch_indices = needs_response[batch_start : batch_start + batch_size]
        tasks = []
        for idx in batch_indices:
            synth = records[idx].get("synthetic_data") or records[idx]
            instruction = str(synth.get("instruction", "")).strip()
            tasks.append(generate_response(llm, instruction, semaphore))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, resp in zip(batch_indices, responses):
            if isinstance(resp, Exception):
                logger.warning("Error generating response for record %d: %s", idx, resp)
                response_text = ""
            else:
                response_text = str(resp).strip()

            if "synthetic_data" in records[idx]:
                records[idx]["synthetic_data"]["response"] = response_text
            else:
                records[idx]["response"] = response_text

        done += len(batch_indices)
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(needs_response) - done) / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d (%.0f/s, ETA %.0fs)",
            done, len(needs_response), rate, eta,
        )

    # Write back
    logger.info("Writing updated records to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    with_response = sum(
        1 for rec in records
        if str((rec.get("synthetic_data") or rec).get("response", "")).strip()
    )
    logger.info(
        "Done: %d/%d records now have responses (%.1f%%)",
        with_response, len(records), 100.0 * with_response / len(records),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SynEval Step 1b: Generate responses for evolved instructions."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to synthesized JSONL (output of 01_synthesize.py).")
    parser.add_argument("--output-dir", type=str, default="syneval/data",
                        help="Output directory. Default: same as input.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output filename. Default: overwrites input file.")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8100/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--concurrency", type=int, default=100,
                        help="Max concurrent LLM requests (default: 100).")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for progress reporting (default: 500).")
    parser.add_argument("--rpm-limit", type=int, default=0)
    parser.add_argument("--tpm-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(generate_all(args))


if __name__ == "__main__":
    main()
