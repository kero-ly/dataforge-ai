# DataForge

**High-concurrency async pipeline for LLM training data synthesis and curation.**

[![CI](https://github.com/kero-ly/dataforge/actions/workflows/ci.yml/badge.svg)](https://github.com/kero-ly/dataforge/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dataforge)](https://pypi.org/project/dataforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[中文文档](README_zh.md)

---

## Table of Contents

- [What is DataForge?](#what-is-dataforge)
- [Before & After](#before--after)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [CLI Overview](#cli-overview)
- [Supported Backends](#supported-backends)
- [Configuration Reference](#configuration-reference)
- [Key Features](#key-features)
- [Benchmarks](#benchmarks)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## What is DataForge?

Building SFT and RLHF datasets at scale is an infrastructure problem before it is a data problem. Naive async scripts hit rate limits silently, crash without checkpoints, produce malformed JSON, and leave no audit trail. At 10K+ records these failure modes compound: a crash at 80% means losing hours of API spend and starting over.

DataForge is a production-grade async pipeline that handles rate limiting, retries, checkpointing, and quality filtering automatically — so you focus on *what* to generate, not *how*. It works the same way whether you are hitting a cloud API or a local vLLM server.

**Who it is for:** researchers and engineers running WizardLM-style instruction evolution, building SFT datasets from seed corpora, filtering and scoring synthetic data at scale (1K–100K+ records), or comparing data quality across model providers.

---

## Before & After

Here is what DataForge does to a seed instruction using `evol-instruct` at depth 2:

**Input** (`seeds.jsonl`):
```json
{"id": "q1", "instruction": "What is machine learning?"}
```

**Output** (`output.jsonl`) after two rounds of WizardLM-style mutation:
```json
{
  "id": "q1",
  "seed_data": {"instruction": "What is machine learning?"},
  "synthetic_data": {
    "instruction": "You are designing a production ML system for a fintech company with strict latency requirements (<50ms p99). Explain how machine learning works, covering: (1) the mathematical relationship between features and predictions, (2) how models generalize beyond training data, and (3) two concrete failure modes that arise when deploying in low-latency environments."
  },
  "score": 4.5,
  "status": "COMPLETED"
}
```

EvolInstruct applies WizardLM-style mutations — add constraints, deepen complexity, concretize context — to make instructions harder and richer. The `depth` field controls how many mutation rounds are applied per record.

---

## Installation

```bash
pip install dataforge
```

**Requirements:** Python 3.10+. No GPU needed for cloud API mode.

For local LLM inference (vLLM / Ollama):
```bash
pip install vllm   # requires a CUDA-capable GPU
```

Install from source:
```bash
git clone https://github.com/kero-ly/dataforge.git
cd dataforge
pip install -e ".[dev]"
```

---

## Quick Start

### Try it now (no API key)

The fastest way to see DataForge in action — no credentials required.

**Step 1.** Create `seeds.jsonl`:
```json
{"id": "q1", "instruction": "What is machine learning?"}
{"id": "q2", "instruction": "Explain the attention mechanism in Transformers."}
{"id": "q3", "instruction": "What is overfitting and how do you prevent it?"}
```

**Step 2.** Create `config.yaml`:
```yaml
name: quickstart
source:
  type: jsonl
  path: ./seeds.jsonl
pipeline:
  - step: generate
    strategy: evol-instruct
    depth: 2
sink:
  path: ./output.jsonl
concurrency: 4
```

**Step 3.** Run:
```bash
python -m dataforge run config.yaml --backend fake
```

Expected output:
```
Pipeline completed: 3/3 records
  Completed: 3  Rejected: 0  Failed: 0
  Elapsed: 0.1s  Throughput: 30.0 rec/s
```

Results are written to `output.jsonl`. Each record contains `seed_data`, `synthetic_data`, `score`, and `status`.

---

### With a cloud API (OpenAI / DeepSeek)

**Step 1.** Use the same `seeds.jsonl` from above.

**Step 2.** Create `config.yaml`:
```yaml
name: cloud-pipeline
source:
  type: jsonl
  path: ./seeds.jsonl
pipeline:
  - step: generate
    strategy: evol-instruct
    depth: 3
    mutation_types: [constraints, deepen, concretize]
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}   # reads from environment
      rpm_limit: 60
      tpm_limit: 100000
      generation_kwargs:
        temperature: 0.7
        max_tokens: 1024
  - step: evaluate
    evaluator: llm-judge
    criteria: helpfulness
    threshold: 4.0
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
sink:
  path: ./output.jsonl
  checkpoint_dir: ./.dataforge_runs   # crash-safe: restart with same command
  dead_letter_path: ./failed.jsonl    # rejected records saved here
concurrency: 20
```

**Step 3.** Run:
```bash
export OPENAI_API_KEY=sk-...
dataforge run config.yaml
```

> **DeepSeek:** add `base_url: https://api.deepseek.com/v1` and use `api_key: ${DEEPSEEK_API_KEY}`. Everything else stays the same.

If the run is interrupted, restart with the exact same command — DataForge reads the checkpoint and skips already-completed records automatically.

---

### With a local vLLM server

**Step 1.** Use the same `seeds.jsonl` from above.

**Step 2.** Create `config.yaml`:
```yaml
name: local-pipeline
source:
  type: jsonl
  path: ./seeds.jsonl
pipeline:
  - step: generate
    strategy: evol-instruct
    depth: 3
    mutation_types: [constraints, deepen, concretize]
    llm:
      provider: vllm
      model: Qwen/Qwen2.5-7B-Instruct
      base_url: http://localhost:8000/v1
      concurrency: 50
  - step: evaluate
    evaluator: regex-filter
    blacklist_patterns: ["I cannot", "I'm sorry", "As an AI"]
sink:
  path: ./output.jsonl
  checkpoint_dir: ./.dataforge_runs
concurrency: 50
```

**Step 3.** Start vLLM and run:
```bash
# Terminal 1
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# Terminal 2
dataforge run config.yaml
```

---

## How It Works

```
seeds.jsonl → [Producer] → asyncio.Queue → [Worker ×N] → output.jsonl
                                                │
                                       strategy.apply()    ← evolves instruction
                                                │
                                      [evaluator chain]    ← filter / score
                                                │
                                   checkpoint.commit()     ← WAL append
```

`N` workers pull records from a bounded queue. Each worker runs the strategy (EvolInstruct mutates the instruction across `depth` rounds), then passes the result through the evaluator chain (RegexFilter rejects blacklisted patterns; LLMJudge scores 1–5 and rejects below threshold). Only records that pass all evaluators are written to the output JSONL. After each write, the record ID is appended to a write-ahead log — crash and restart with the same command, zero records lost.

The dual-bucket rate limiter enforces both RPM and TPM simultaneously on every LLM client, preventing silent quota violations that naive async code causes.

---

## CLI Overview

| Command | Description |
|---------|-------------|
| `dataforge run config.yaml` | Run the synthesis pipeline |
| `dataforge run config.yaml --dry-run` | Validate config without running |
| `dataforge assess output.jsonl` | Generate data quality report (JSON / HTML) |
| `dataforge benchmark config.yaml` | Run MT-Bench / IF-Eval lite evaluation |
| `dataforge status .dataforge_runs` | Show checkpoint progress |
| `dataforge inspect output.jsonl` | Print output statistics |

---

## Supported Backends

| Provider | `provider` value | Notes |
|----------|-----------------|-------|
| OpenAI | `openai` | GPT-4o, GPT-4o-mini, o1, etc. |
| Anthropic | `anthropic` | Claude 3.x via OpenAI-compatible endpoint |
| DeepSeek | `openai` | Set `base_url: https://api.deepseek.com/v1` |
| vLLM | `vllm` | Self-hosted (Qwen, Llama, Mistral, Phi…) |
| Ollama | `vllm` | Set `base_url: http://localhost:11434/v1` |
| Alibaba Bailian | `bailian` | 百炼 DashScope API |

Switch providers by changing two lines in `config.yaml` — no code changes required.

---

## Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `source.path` | — | Input path (JSONL / CSV / Parquet) |
| `sink.path` | — | Output JSONL path |
| `sink.checkpoint_dir` | `.dataforge_runs` | WAL checkpoint directory |
| `sink.dead_letter_path` | `null` | Path for rejected / failed records |
| `concurrency` | `50` | Max concurrent LLM requests |
| `mode` | `streaming` | `streaming` (large datasets) or `burst` (fits in RAM) |
| `pipeline[].step` | — | `generate` or `evaluate` |
| `pipeline[].strategy` | `evol-instruct` | `evol-instruct` · `paraphrase` · `seed-to-qa` · `self-play` |
| `pipeline[].depth` | `3` | EvolInstruct mutation rounds |
| `pipeline[].mutation_types` | all | `constraints` · `deepen` · `concretize` |
| `pipeline[].evaluator` | — | `regex-filter` · `llm-judge` · `length-filter` |
| `llm.provider` | — | See [Supported Backends](#supported-backends) |
| `llm.rpm_limit` | `60` | Requests-per-minute bucket |
| `llm.tpm_limit` | `100000` | Tokens-per-minute bucket |
| `llm.api_key` | — | Supports `${ENV_VAR}` interpolation |
| `llm.generation_kwargs` | `{}` | Passed to LLM: `temperature`, `max_tokens`, etc. |

---

## Key Features

| Feature | Description |
|---------|-------------|
| Async-first | Single bounded queue + N-worker pattern; non-blocking I/O throughout |
| Dual rate limiting | RPM + TPM token buckets with continuous refill |
| WAL checkpointing | Crash-safe; SQLite and Redis backends available |
| EvolInstruct | WizardLM-style mutation: constraints / deepen / concretize |
| LLMJudge | 1–5 scoring evaluator with regex-first + fallback digit extraction |
| RegexFilter | Blacklist pattern matching + optional JSON schema validation |
| Distributed | Ray Actor, Dask, and sharding backends |
| Unified routing | One config works for OpenAI, Anthropic, DeepSeek, vLLM, Ollama |
| Data quality CLI | `dataforge assess` generates structured quality reports |

---

## Benchmarks

All experiments run on Qwen2.5-7B-Instruct unless noted. Statistical tests use two-sided t-test.

### Throughput

| Baseline | DataForge | Improvement |
|----------|-----------|-------------|
| Sequential (1 worker) | 2,419 rec/min | **48×** |
| Naive Async | 2,419 vs 2,330 rec/min | **+3.82%**, p=0.013, 95% CI=[44.3, 133.6] |
| Distilabel | — | **+152%** throughput |

### Fault Tolerance (30% random failure injection)

| Method | Completion Rate |
|--------|----------------|
| DataForge | **99.9%** |
| Naive Async (no retry) | 70.4% |
| **Advantage** | **+29 pp** |

Consistent across 3 independent experiments (7B and 14B models).

### Cloud API Rate Limit Compliance (qwen-plus, RPM=120)

| Method | Completion |
|--------|-----------|
| DataForge dual-bucket | **89.2%** (446/500) |
| No limiter | 40.0% (200/500) |
| Naive Async | 0.0% (0/500, vLLM crash) |

### Distributed Scaling (RayActor, 4× workers)

| Workers | Throughput | Efficiency |
|---------|-----------|------------|
| 1× | 1,482 rpm | — |
| 2× | 2,798 rpm | 94.4% |
| 4× | **5,361 rpm** | **90.5%** |

---

## Documentation

- [Architecture Overview](docs/plans/structure.md)
- [Configuration Reference](docs/plans/api_interface.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, coding standards, and the PR workflow.

---

## Citation

If you use DataForge in your research, please cite:

```bibtex
@software{dataforge2026,
  title  = {DataForge: High-Concurrency Async Pipeline for LLM Training Data Synthesis},
  year   = {2026},
  url    = {https://github.com/kero-ly/dataforge}
}
```

---

## License

MIT © [DataForge Contributors](https://github.com/kero-ly/dataforge/graphs/contributors)
