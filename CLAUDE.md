# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (skip slow exhaustion test)
pytest tests/ -k "not rpm_exhaustion"

# Run a single test file
pytest tests/test_pipeline.py

# Lint
ruff check .

# Type check
mypy src/dataforge/

# Coverage (minimum 70% required)
pytest tests/ --cov=src/dataforge --cov-report=html

# CLI usage
dataforge run config.yaml [--dry-run]

# Demo script (offline, no API key needed)
python main.py --backend fake
```

## Architecture

DataForge is an async pipeline for synthesizing and curating LLM training data. The core flow is:

```
Input JSONL → Producer → asyncio.Queue → Worker Pool → Output JSONL
                                               ↓
                                         Strategy.apply()      ← generates synthetic data
                                               ↓
                                      [Evaluator chain]        ← filters/scores records
                                               ↓
                                     Checkpoint.commit()       ← WAL append to .jsonl
```

### Key modules and their roles

**`pipeline.py`** — Orchestrator. Implements the single bounded queue + N-worker pattern. On startup it reads `checkpoint.jsonl` to skip already-completed record IDs. Progress is rendered via Rich (auto-disabled in non-TTY environments).

**`schema.py`** — `DataRecord` (Pydantic) is the single unit flowing through the pipeline. `RecordStatus` tracks lifecycle: `PENDING → GENERATED → EVALUATING → COMPLETED | REJECTED | FAILED`.

**`config/`** — `ForgeConfig` / `LLMConfig` / `GenerateStepConfig` / `EvaluateStepConfig` are Pydantic schemas for YAML configs. `loader.py` parses YAML, resolves `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env vars, and builds a `Pipeline` instance.

**`strategies/`** — `BaseStrategy` has one method: `async def apply(record) -> record`. The built-in `EvolInstruct` runs `depth` mutation iterations (randomly chosen from constraints / deepen / concretize). When `require_json=True`, it attempts regex strip first, then a secondary LLM repair call (max 2 attempts).

**`evaluators/`** — `BaseEvaluator` has one method: `async def evaluate(record) -> bool`. `_get_content()` helper lives on the base class. `RegexFilter` does blacklist pattern matching + optional JSON schema validation. `LLMJudge` prompts an LLM for a 1–5 score; uses regex fullmatch first, falls back to extracting the last valid digit from explanations.

**`clients/`** — `BaseLLMClient` integrates the dual-bucket rate limiter. `OpenAIClient` wraps the async OpenAI SDK (works for OpenAI, Anthropic, DeepSeek). `vLLMClient` extends it for local servers (vLLM, Ollama, SGLang).

**`engine/`** — Three self-contained components:
- `TokenBucketRateLimiter`: dual RPM + TPM buckets with continuous refill
- `RetryEngine`: exponential backoff with full jitter; retries strategy failures only (not evaluator rejections)
- `CheckpointManager`: WAL appends after successful processing; silently skips corrupt lines on load

### Extension points

To add a custom strategy: subclass `BaseStrategy`, implement `async def apply(record: DataRecord) -> DataRecord`.

To add a custom evaluator: subclass `BaseEvaluator`, implement `async def evaluate(record: DataRecord) -> bool`.

Public API (importable from `dataforge`): `Pipeline`, `DataRecord`, `RecordStatus`, `EvolInstruct`, `LLMJudge`, `RegexFilter`.
