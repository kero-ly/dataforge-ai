# Changelog

All notable changes to DataForge are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-04-17

### Added

**Core Pipeline**
- Async pipeline with single bounded queue + N-worker pattern (`pipeline.py`)
- `DataRecord` (Pydantic) as the single unit flowing through the pipeline
- `RecordStatus` lifecycle: `PENDING → GENERATED → EVALUATING → COMPLETED | REJECTED | FAILED`
- Rich progress bars (auto-disabled in non-TTY environments)
- Metadata and lineage tracking on every record (timestamp, token count, model version, retry count)

**Strategies**
- `EvolInstruct` — WizardLM-style instruction evolution with 3 mutation types: constraints, deepen, concretize
- `Paraphrase`, `SeedToQA`, `SelfPlay` strategies
- JSON self-repair: regex strip + secondary LLM repair call (max 2 attempts) when `require_json=True`

**Evaluators**
- `LLMJudge` — 1–5 scoring; regex fullmatch first, fallback to last valid digit extraction
- `RegexFilter` — blacklist pattern matching + optional JSON schema validation
- `LengthFilter`, `LengthWindow`, `MultiCriteria`, `Similarity`, `Completeness` evaluators

**Engine**
- `TokenBucketRateLimiter` — dual RPM + TPM buckets with continuous refill
- `RetryEngine` — exponential backoff with full jitter; retries strategy failures only
- `CheckpointManager` — WAL append after successful processing; SQLite and Redis backends

**Clients**
- `OpenAIClient` — async OpenAI SDK wrapper (works for OpenAI, Anthropic, DeepSeek)
- `vLLMClient` — extends OpenAIClient for local servers (vLLM, Ollama, SGLang)
- `BailianClient` — Alibaba Cloud 百炼 API
- `FallbackClient` — automatic failover across multiple backends

**Distributed**
- Ray Actor backend (`RayOrchestrator`) — persistent event loop per actor, no cold start
- Dask backend
- Sharding backend

**CLI**
- `dataforge run config.yaml [--dry-run]` — YAML-driven pipeline execution
- `dataforge assess` — data quality reporting (JSON/Markdown/HTML)
- `dataforge benchmark` — MT-Bench lite, IF-Eval lite, Safety lite

**Assessment & Benchmark**
- Data quality reporting with SFT readiness suite
- Built-in benchmark task sets

### Performance

- 48× throughput vs sequential baseline
- +3.82% vs Naive Async (p=0.013, 95% CI=[44.3, 133.6])
- 99.9% completion rate under 30% fault injection vs 70.4% for unprotected async
- Linear distributed scaling: 5,361 rpm at 4× workers (90.5% efficiency)
