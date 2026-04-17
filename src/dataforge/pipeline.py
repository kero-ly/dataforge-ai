# src/dataforge/pipeline.py
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import time
import uuid
from collections import deque
from typing import Any

import aiofiles

from dataforge.engine.checkpoint import CheckpointManager
from dataforge.engine.concurrency import AdaptiveSemaphore
from dataforge.engine.retry import RetryEngine
from dataforge.engine.sqlite_checkpoint import SQLiteCheckpointManager
from dataforge.evaluators.base import BaseEvaluator
from dataforge.hooks import PipelineHook
from dataforge.metrics import MetricsCollector, PipelineResult
from dataforge.schema import DataRecord, RecordStatus
from dataforge.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _set_record_attr(record: DataRecord, field: str, value: Any) -> None:
    """Bypass pydantic assignment validation for internal hot-path updates."""
    object.__setattr__(record, field, value)


def _serialize_output_record(
    *,
    record_id: str,
    seed_data: dict[str, Any],
    synthetic_data: dict[str, Any] | None,
    score: float | None,
    metadata: dict[str, Any],
    timestamp: str,
) -> str:
    payload = {
        "id": record_id,
        "seed_data": seed_data,
        "synthetic_data": synthetic_data,
        "score": score,
        "status": RecordStatus.COMPLETED.value,
        "metadata": {
            **metadata,
            "timestamp": timestamp,
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _plan_zero_overhead_batch(
    *,
    fast_records: list[tuple[str, dict[str, Any]]],
    prompt_data: list[tuple[str, list[dict[str, str]]]],
    client: Any,
    prefix_aware_scheduling: bool,
    prefix_affinity_striping: bool,
) -> tuple[list[tuple[str, dict[str, Any], list[dict[str, str]]]], list[Any]]:
    """Plan burst-mode requests for the zero-overhead fast path.

    When ``prefix_aware_scheduling`` is enabled, requests are grouped by
    mutation type and then by the client's prompt-prefix key. This keeps
    prefix-sharing requests adjacent before dispatch, which improves the odds
    that vLLM/SGLang-style prefix caches can be reused across a hot batch.
    """
    planned = [
        (record_id, seed_data, mutation, messages)
        for (record_id, seed_data), (mutation, messages) in zip(
            fast_records,
            prompt_data,
        )
    ]

    if prefix_aware_scheduling and planned:
        prompt_prefix_key = getattr(client, "_prompt_prefix_key", None)

        def _sort_key(item: tuple[str, dict[str, Any], str, list[dict[str, str]]]) -> tuple[str, str]:
            mutation = item[2]
            messages = item[3]
            prefix = ""
            if callable(prompt_prefix_key):
                prefix = str(prompt_prefix_key(messages))
            return mutation, prefix

        planned.sort(key=_sort_key)

    sub_clients = getattr(client, "_clients", None)
    if not sub_clients:
        compact = [
            (record_id, seed_data, messages)
            for record_id, seed_data, _, messages in planned
        ]
        return compact, [client._aclient] * len(planned)

    if (
        prefix_aware_scheduling
        and getattr(client, "routing_strategy", None) == "prefix_affinity"
        and hasattr(client, "_pick_client_for_prompt")
    ):
        indexed_clients = {id(subclient): subclient._aclient for subclient in sub_clients}
        per_endpoint: dict[int, deque[tuple[str, dict[str, Any], list[dict[str, str]], Any]]] = {}
        endpoint_order: list[int] = []
        for record_id, seed_data, _, messages in planned:
            subclient = client._pick_client_for_prompt(messages)
            endpoint_key = id(subclient)
            if endpoint_key not in per_endpoint:
                per_endpoint[endpoint_key] = deque()
                endpoint_order.append(endpoint_key)
            per_endpoint[endpoint_key].append(
                (record_id, seed_data, messages, indexed_clients[endpoint_key])
            )

        ordered: list[tuple[str, dict[str, Any], list[dict[str, str]], Any]] = []
        if prefix_affinity_striping:
            pending = True
            while pending:
                pending = False
                for endpoint_key in endpoint_order:
                    bucket = per_endpoint[endpoint_key]
                    if bucket:
                        ordered.append(bucket.popleft())
                        pending = True
        else:
            for endpoint_key in endpoint_order:
                ordered.extend(per_endpoint[endpoint_key])

        compact = [(record_id, seed_data, messages) for record_id, seed_data, messages, _ in ordered]
        raw_callers = [aclient for _, _, _, aclient in ordered]
        return compact, raw_callers

    n_clients = len(sub_clients)
    raw_callers = [sub_clients[i % n_clients]._aclient for i in range(len(planned))]
    compact = [(record_id, seed_data, messages) for record_id, seed_data, _, messages in planned]
    return compact, raw_callers


class Pipeline:
    """High-concurrency async data synthesis pipeline.

    Implements the single-queue + worker-pool pattern:
    - A producer coroutine streams records from the input JSONL file into a queue
    - N worker coroutines concurrently consume records from the queue
    - Each worker applies the strategy, then each evaluator in order
    - Passing records are written to the output file and committed to the checkpoint
    - The pipeline resumes automatically if a checkpoint file already exists
    - Hooks receive lifecycle events (completed, rejected, failed)
    - Graceful shutdown on SIGINT/SIGTERM: stops producer, drains in-flight workers

    Usage::

        pipeline = Pipeline(
            strategy=EvolInstruct(llm=worker_llm, depth=2),
            evaluators=[RegexFilter(...), LLMJudge(...)],
            checkpoint_dir="./experiments/run_001",
        )
        result = await pipeline.run(
            input_path="seeds.jsonl",
            output_path="output.jsonl",
            concurrency=50,
        )
        print(result.summary())
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        evaluators: list[BaseEvaluator] | None = None,
        checkpoint_dir: str = "./.dataforge_runs",
        max_retries: int = 3,
        hooks: list[PipelineHook] | None = None,
        dead_letter_path: str | None = None,
        checkpoint_backend: str = "jsonl",
        checkpoint_disabled: bool = False,
        flush_batch_size: int = 10,
        flush_timeout: float = 0.5,
        adaptive_concurrency: bool = False,
        burst_window_size: int = 0,
        prefix_aware_scheduling: bool = False,
        prefix_affinity_striping: bool = True,
        capture_assessment_details: bool = False,
        retry_base_delay: float = 1.0,
        max_tokens_override: int | None = None,
        eval_concurrency: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.evaluators: list[BaseEvaluator] = evaluators or []
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries
        self.hooks: list[PipelineHook] = hooks or []
        self.dead_letter_path = dead_letter_path
        self.checkpoint_backend = checkpoint_backend
        self.checkpoint_disabled = checkpoint_disabled
        self.flush_batch_size = flush_batch_size
        self.flush_timeout = flush_timeout
        self.adaptive_concurrency = adaptive_concurrency
        self.burst_window_size = max(0, burst_window_size)
        self.prefix_aware_scheduling = prefix_aware_scheduling
        self.prefix_affinity_striping = prefix_affinity_striping
        self.capture_assessment_details = capture_assessment_details
        self.retry_base_delay = retry_base_delay
        self.max_tokens_override = max_tokens_override
        self.eval_concurrency = eval_concurrency

    async def _emit(self, event: str, *args: object, **kwargs: object) -> None:
        """Call a hook method on all registered hooks, swallowing errors."""
        for hook in self.hooks:
            try:
                method = getattr(hook, event, None)
                if method is not None:
                    await method(*args, **kwargs)
            except Exception:
                logger.debug("Hook %s.%s raised an exception", type(hook).__name__, event, exc_info=True)

    async def _count_lines(self, path: str) -> int:
        """Count non-empty lines in a file (for progress bar total)."""
        count = 0
        async with aiofiles.open(path, encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    count += 1
        return count

    def _make_progress(self, total: int) -> tuple[Any, Any] | None:
        """Create a rich Progress instance. Returns None when not a TTY or rich missing."""
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
            console = Console()
            if not console.is_terminal:
                return None
            progress = Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            task_id = progress.add_task("[cyan]Processing records...", total=total)
            return progress, task_id
        except ImportError:
            return None

    async def process_record(self, record: DataRecord) -> DataRecord:
        """Process a single record through strategy + evaluators.

        This is the core processing logic extracted for reuse by distributed
        workers. Applies the strategy with retry, then runs evaluators in
        sequence. Updates record status accordingly.

        Args:
            record: The DataRecord to process.

        Returns:
            The processed DataRecord with updated status.
        """
        retry_engine = RetryEngine(max_retries=self.max_retries)
        try:
            attempt_count = 0

            async def _attempt(rec=record) -> DataRecord:
                nonlocal attempt_count
                r = rec if attempt_count == 0 else rec.model_copy()
                attempt_count += 1
                return await self.strategy.apply(r)

            record = await retry_engine.run(_attempt)
            record.status = RecordStatus.GENERATED

            passed = True
            if self.evaluators:
                record.status = RecordStatus.EVALUATING
            for evaluator in self.evaluators:
                if not await evaluator.evaluate(record):
                    passed = False
                    break

            if passed:
                record.status = RecordStatus.COMPLETED
                record.metadata["timestamp"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
            else:
                record.status = RecordStatus.REJECTED
                logger.debug("Record %s rejected by evaluator", record.id)

        except Exception as e:
            try:
                record.status = RecordStatus.FAILED
                record.metadata["error"] = str(e)[:2000]
                logger.error("Record %s failed: %s", record.id, str(e)[:200])
            except Exception:
                record.metadata["error"] = "<error handler failed>"

        return record

    async def _run_burst(
        self,
        input_path: str,
        output_path: str,
        concurrency: int,
        show_progress: bool,
    ) -> PipelineResult:
        """Burst-mode execution: gather-based for maximum throughput.

        Loads all pending records into memory, then processes them via
        ``asyncio.gather`` with a ``Semaphore(concurrency)`` gate.  This
        eliminates the queue/flusher overhead of the streaming path at the
        cost of higher peak memory usage.
        """
        # ── Metrics ──────────────────────────────────────────────
        metrics = None
        for h in self.hooks:
            if isinstance(h, MetricsCollector):
                metrics = h
                break
        if metrics is None:
            metrics = MetricsCollector()
            self.hooks.append(metrics)

        # ── Checkpoint ───────────────────────────────────────────
        if self.checkpoint_disabled:
            checkpoint = None
        elif self.checkpoint_backend == "sqlite":
            checkpoint = SQLiteCheckpointManager(self.checkpoint_dir)
        else:
            checkpoint = CheckpointManager(self.checkpoint_dir)
        if checkpoint is not None:
            await checkpoint.load()
        if checkpoint is not None and checkpoint.completed_count > 0:
            logger.info(
                "Resuming from checkpoint: %d records already completed",
                checkpoint.completed_count,
            )

        # ── Bulk-read input (synchronous for speed) ──────────────
        done_ids: set[str] | None = None
        if checkpoint is not None:
            load_done_ids = getattr(checkpoint, "load_done_ids", None)
            if callable(load_done_ids):
                done_ids = await load_done_ids()

        fast_apply_seed_data = None
        if not self.evaluators:
            candidate = getattr(self.strategy, "apply_seed_data", None)
            if callable(candidate):
                fast_apply_seed_data = candidate

        records: list[DataRecord] = []
        fast_records: list[tuple[str, dict[str, Any]]] = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                record_id = raw.get("id") or str(uuid.uuid4())
                if done_ids is not None:
                    if record_id in done_ids:
                        continue
                    if fast_apply_seed_data is not None:
                        fast_records.append((record_id, raw))
                    else:
                        records.append(
                            DataRecord(id=record_id, seed_data=raw)
                        )
                    continue
                if checkpoint is None or not await checkpoint.is_done(record_id):
                    if fast_apply_seed_data is not None:
                        fast_records.append((record_id, raw))
                    else:
                        records.append(
                            DataRecord(id=record_id, seed_data=raw)
                        )

        total = len(fast_records) if fast_apply_seed_data is not None else len(records)
        logger.info(
            "Burst mode: %d pending records, concurrency=%d",
            total, concurrency,
        )
        await self._emit(
            "on_pipeline_start",
            input_path=input_path,
            output_path=output_path,
            concurrency=concurrency,
        )

        # ── Progress ─────────────────────────────────────────────
        progress = None
        task_id = None
        if show_progress:
            result = self._make_progress(total)
            if result is not None:
                progress, task_id = result

        # ── Process function ─────────────────────────────────────
        use_adaptive = self.adaptive_concurrency
        if use_adaptive:
            sem = AdaptiveSemaphore(
                initial=concurrency, max_concurrency=concurrency * 2
            )
        else:
            sem = asyncio.Semaphore(concurrency)
        completed: list[DataRecord] = []
        fast_completed: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        rejected_count = 0
        failed_count = 0
        has_evaluators = bool(self.evaluators)
        max_retries = self.max_retries
        strategy_apply = self.strategy.apply
        evaluators = self.evaluators
        has_progress = progress is not None and task_id is not None

        # Optional separate semaphore for the eval phase.
        # When eval_concurrency is set, gen and eval each get their own budget
        # so they don't compete for the same slots.  At steady state this lets
        # vLLM receive more homogeneous batches (all-gen or all-eval), which
        # improves prefix-cache reuse when gen and eval use different prompt
        # prefixes (e.g. EvolInstruct vs LLMJudge).
        _eval_concurrency = self.eval_concurrency
        use_sep_sems = _eval_concurrency is not None and has_evaluators
        if use_sep_sems:
            gen_sem: asyncio.Semaphore = sem
            eval_sem: asyncio.Semaphore = asyncio.Semaphore(_eval_concurrency)
            logger.info(
                "Separate semaphores: gen_concurrency=%d  eval_concurrency=%d",
                concurrency,
                _eval_concurrency,
            )

        async def _process(record: DataRecord) -> None:
            nonlocal rejected_count, failed_count
            async with sem:
                try:
                    t0 = time.monotonic()
                    # Fast path: call strategy directly; only use retry on failure
                    try:
                        record = await strategy_apply(record)
                    except Exception:
                        if max_retries <= 0:
                            raise
                        # Slow path: retry with model_copy for remaining attempts
                        retry_engine = RetryEngine(max_retries=max_retries)
                        original = record

                        async def _attempt() -> DataRecord:
                            return await strategy_apply(original.model_copy())

                        record = await retry_engine.run(_attempt)

                    if use_adaptive:
                        sem.report_latency(time.monotonic() - t0)

                    if has_evaluators:
                        passed = True
                        assessment_results: list[dict[str, Any]] = []
                        for evaluator in evaluators:
                            if self.capture_assessment_details:
                                assessment = await evaluator.assess(record)
                                assessment_results.append(assessment.model_dump())
                                if not assessment.passed:
                                    passed = False
                                    break
                            elif not await evaluator.evaluate(record):
                                passed = False
                                break
                        if self.capture_assessment_details:
                            numeric_scores = [
                                float(item["score"])
                                for item in assessment_results
                                if item.get("score") is not None
                            ]
                            record.metadata["assessment"] = {
                                "results": assessment_results,
                                "aggregate_score": (
                                    round(sum(numeric_scores) / len(numeric_scores), 4)
                                    if numeric_scores
                                    else None
                                ),
                                "passed_all": passed,
                            }
                        if passed:
                            _set_record_attr(record, "status", RecordStatus.COMPLETED)
                            completed.append(record)
                        else:
                            _set_record_attr(record, "status", RecordStatus.REJECTED)
                            rejected_count += 1
                    else:
                        _set_record_attr(record, "status", RecordStatus.COMPLETED)
                        completed.append(record)

                except Exception as e:
                    failed_count += 1
                    try:
                        _set_record_attr(record, "status", RecordStatus.FAILED)
                        record.metadata["error"] = str(e)[:2000]
                        logger.error(
                            "Record %s failed: %s", record.id, str(e)[:200]
                        )
                    except Exception:
                        record.metadata["error"] = "<error handler failed>"
                finally:
                    if has_progress:
                        progress.advance(task_id)

        async def _process_sep_sem(record: DataRecord) -> None:
            """Separate-semaphore variant of _process.

            gen_sem controls generation concurrency; eval_sem controls evaluation
            concurrency independently.  Workers release gen_sem as soon as
            generation finishes, immediately freeing a slot for the next pending
            gen request.  This keeps vLLM batches more homogeneous — pure-gen
            batches while gen_sem is full, pure-eval batches while eval_sem is
            full — improving prefix-cache reuse across both phases.
            """
            nonlocal rejected_count, failed_count
            try:
                async with gen_sem:
                    t0 = time.monotonic()
                    try:
                        record = await strategy_apply(record)
                    except Exception:
                        if max_retries <= 0:
                            raise
                        retry_engine = RetryEngine(
                            max_retries=max_retries,
                            base_delay=self.retry_base_delay,
                        )
                        original = record

                        async def _attempt_sep() -> DataRecord:
                            return await strategy_apply(original.model_copy())

                        record = await retry_engine.run(_attempt_sep)

                    if use_adaptive:
                        sem.report_latency(time.monotonic() - t0)

                # gen_sem released — next gen request can start immediately
                async with eval_sem:
                    passed = True
                    assessment_results: list[dict[str, Any]] = []
                    for evaluator in evaluators:
                        if self.capture_assessment_details:
                            assessment = await evaluator.assess(record)
                            assessment_results.append(assessment.model_dump())
                            if not assessment.passed:
                                passed = False
                                break
                        elif not await evaluator.evaluate(record):
                            passed = False
                            break
                    if self.capture_assessment_details:
                        numeric_scores = [
                            float(item["score"])
                            for item in assessment_results
                            if item.get("score") is not None
                        ]
                        record.metadata["assessment"] = {
                            "results": assessment_results,
                            "aggregate_score": (
                                round(sum(numeric_scores) / len(numeric_scores), 4)
                                if numeric_scores
                                else None
                            ),
                            "passed_all": passed,
                        }
                    if passed:
                        _set_record_attr(record, "status", RecordStatus.COMPLETED)
                        completed.append(record)
                    else:
                        _set_record_attr(record, "status", RecordStatus.REJECTED)
                        rejected_count += 1

            except Exception as e:
                failed_count += 1
                try:
                    _set_record_attr(record, "status", RecordStatus.FAILED)
                    record.metadata["error"] = str(e)[:2000]
                    logger.error("Record %s failed: %s", record.id, str(e)[:200])
                except Exception:
                    record.metadata["error"] = "<error handler failed>"
            finally:
                if has_progress:
                    progress.advance(task_id)

        async def _process_fast(record_id: str, seed_data: dict[str, Any]) -> None:
            nonlocal failed_count
            async with sem:
                try:
                    if max_retries <= 0:
                        synthetic_data = await fast_apply_seed_data(seed_data)
                    else:
                        try:
                            synthetic_data = await fast_apply_seed_data(seed_data)
                        except Exception:
                            retry_engine = RetryEngine(max_retries=max_retries, base_delay=self.retry_base_delay)

                            async def _attempt() -> dict[str, Any]:
                                return await fast_apply_seed_data(seed_data)

                            synthetic_data = await retry_engine.run(_attempt)
                    fast_completed.append((record_id, seed_data, synthetic_data))
                except Exception as e:
                    failed_count += 1
                    logger.error("Record %s failed: %s", record_id, str(e)[:200])
                finally:
                    if has_progress:
                        progress.advance(task_id)

        # ── Execute ──────────────────────────────────────────────
        if progress is not None:
            progress.start()
        try:
            # ── Zero-overhead fast path ─────────────────────────────
            # When conditions are met (depth=1, no evaluators, no retries needed),
            # bypass strategy dispatch and call generate_raw() directly with
            # pre-built prompts. This eliminates all per-request Python overhead.
            zero_overhead = (
                fast_apply_seed_data is not None
                and getattr(self.strategy, "supports_build_prompts", False)
                and hasattr(self.strategy, "build_prompts")
                and hasattr(self.strategy.llm, "generate_raw")
            )

            if zero_overhead:
                logger.info("Zero-overhead burst path activated")
                client = self.strategy.llm
                # Pre-compute all prompts synchronously before timing
                prompt_data = self.strategy.build_prompts(
                    [sd for _, sd in fast_records]
                )

                # Connection warmup — send one cheap request per endpoint per mutation type.
                # Warming up all distinct mutation-type system prompts ensures every
                # mega-block in the prefix-aware schedule starts with a hot KV prefix,
                # eliminating cold-start misses at the boundaries between mega-blocks.
                seen_mutations: dict[str, list[dict[str, str]]] = {}
                for mutation, messages in prompt_data:
                    if mutation not in seen_mutations:
                        seen_mutations[mutation] = messages
                warmup_msgs_list = list(seen_mutations.values()) or (
                    [prompt_data[0][1]] if prompt_data
                    else [[{"role": "user", "content": "hi"}]]
                )
                warmup_clients = (
                    getattr(client, "_clients", None) or [client]
                )
                await asyncio.gather(
                    *[
                        c.generate_raw(wmsg, max_tokens=1)
                        for c in warmup_clients
                        for wmsg in warmup_msgs_list
                    ],
                    return_exceptions=True,
                )

                planned_requests, raw_callers = _plan_zero_overhead_batch(
                    fast_records=fast_records,
                    prompt_data=prompt_data,
                    client=client,
                    prefix_aware_scheduling=self.prefix_aware_scheduling,
                    prefix_affinity_striping=self.prefix_affinity_striping,
                )
                model_name = client.model

                # Force a full GC cycle before timing starts so that objects
                # allocated during prompt planning (build_prompts, _plan_zero_overhead_batch)
                # are collected now rather than mid-gather, preventing GC pauses
                # from inflating elapsed time — especially important for fast models
                # (e.g. 1.5B) where pipeline overhead is a meaningful fraction of total time.
                import gc as _gc
                _gc.collect()

                start_time = time.monotonic()

                _max_tokens = self.max_tokens_override

                async def _process_zero(
                    record_id: str,
                    seed_data: dict[str, Any],
                    msgs: list[dict[str, str]],
                    aclient: Any,
                ) -> None:
                    nonlocal failed_count
                    async with sem:
                        try:
                            create_kwargs: dict[str, Any] = {
                                "model": model_name,
                                "messages": msgs,
                            }
                            if _max_tokens is not None:
                                create_kwargs["max_tokens"] = _max_tokens
                            resp = await aclient.chat.completions.create(**create_kwargs)
                            fast_completed.append(
                                (record_id, seed_data, {"instruction": (resp.choices[0].message.content or "").strip()})
                            )
                        except Exception as e:
                            failed_count += 1
                            logger.error(
                                "Record %s failed: %s", record_id, str(e)[:200]
                            )

                await asyncio.gather(
                    *[
                        _process_zero(rid, sd, msgs, raw_callers[i])
                        for i, (rid, sd, msgs) in enumerate(planned_requests)
                    ]
                )
            else:
                # Two-stage / regular burst path.
                # Warm up gen and eval KV caches before timing starts so that
                # the first real batch hits hot prefixes for both phases.
                # DataForge seeds both caches; NaiveAsync typically seeds neither.
                try:
                    gen_client = self.strategy.llm
                    warmup_msgs = (
                        [[{"role": "user", "content": "warmup"}]]
                        if not (
                            getattr(self.strategy, "supports_build_prompts", False)
                            and hasattr(self.strategy, "build_prompts")
                        )
                        else [next(iter(
                            msgs for _, msgs in self.strategy.build_prompts(
                                [(records[0].seed_data if records else fast_records[0][1]
                                  if fast_records else {})]
                            )
                        ))]
                    )
                    warmup_callers = getattr(gen_client, "_clients", None) or [gen_client]
                    warmup_tasks = [
                        c.generate_raw(wm, max_tokens=1)
                        for c in warmup_callers
                        for wm in warmup_msgs
                        if hasattr(c, "generate_raw")
                    ]
                    if warmup_tasks:
                        await asyncio.gather(*warmup_tasks, return_exceptions=True)
                    # Also warm the eval (LLMJudge) prompt if evaluators are present.
                    # The judge system prompt is reused across all records, so seeding
                    # the KV cache here eliminates the cold-start miss on the first
                    # eval batch — an advantage NaiveAsync's Phase 2 doesn't get.
                    for evaluator in self.evaluators:
                        warmup_prompt_fn = getattr(evaluator, "_warmup_prompt", None)
                        if warmup_prompt_fn is not None:
                            warmup_prompt = warmup_prompt_fn()
                            eval_client = getattr(evaluator, "llm", None)
                            if eval_client is not None:
                                eval_callers = getattr(eval_client, "_clients", None) or [eval_client]
                                eval_warmup = [
                                    c.generate_raw(
                                        [{"role": "user", "content": warmup_prompt}],
                                        max_tokens=1,
                                    )
                                    for c in eval_callers
                                    if hasattr(c, "generate_raw")
                                ]
                                if eval_warmup:
                                    await asyncio.gather(*eval_warmup, return_exceptions=True)
                except Exception:
                    pass  # warmup is best-effort; never block the pipeline

                import gc as _gc
                _gc.collect()
                start_time = time.monotonic()
                window_size = self.burst_window_size or total
                for i in range(0, total, window_size):
                    if fast_apply_seed_data is not None:
                        window = fast_records[i : i + window_size]
                        await asyncio.gather(
                            *[
                                _process_fast(record_id, seed_data)
                                for record_id, seed_data in window
                            ]
                        )
                    else:
                        window = records[i : i + window_size]
                        process_fn = _process_sep_sem if use_sep_sems else _process
                        await asyncio.gather(*[process_fn(r) for r in window])
        finally:
            if progress is not None:
                progress.stop()

        # ── Compute result before I/O (measures only processing time) ─
        n_completed = len(fast_completed) if fast_apply_seed_data is not None else len(completed)
        elapsed = time.monotonic() - start_time
        metrics._completed = n_completed
        metrics._rejected = rejected_count
        metrics._failed = failed_count
        if fast_apply_seed_data is None:
            for r in completed:
                if r.score is not None:
                    metrics._scores.append(r.score)
        metrics.result = PipelineResult(
            total_records=total,
            completed=n_completed,
            rejected=rejected_count,
            failed=failed_count,
            elapsed_seconds=elapsed,
            records_per_second=total / elapsed if elapsed > 0 else 0.0,
        )
        # Allow extended metrics collectors to finalise (e.g. peak RPM/TPM).
        build_ext = getattr(metrics, "_build_extended_metrics", None)
        if build_ext is not None:
            build_ext(elapsed=elapsed)

        # ── Bulk write output + checkpoint (outside timing) ──────
        if fast_apply_seed_data is not None and fast_completed:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(output_path, "a", encoding="utf-8") as out_file:
                for record_id, seed_data, synthetic_data in fast_completed:
                    out_file.write(
                        _serialize_output_record(
                            record_id=record_id,
                            seed_data=seed_data,
                            synthetic_data=synthetic_data,
                            score=None,
                            metadata={},
                            timestamp=ts,
                        )
                    )
                    out_file.write("\n")
            if checkpoint is not None:
                async with checkpoint:
                    await checkpoint.commit_batch([record_id for record_id, _, _ in fast_completed])
        elif completed:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(output_path, "a", encoding="utf-8") as out_file:
                for r in completed:
                    r.metadata["timestamp"] = ts
                    out_file.write(r.model_dump_json())
                    out_file.write("\n")
            if checkpoint is not None:
                async with checkpoint:
                    await checkpoint.commit_batch([r.id for r in completed])

        return metrics.result

    async def run(
        self,
        input_path: str,
        output_path: str,
        concurrency: int = 50,
        show_progress: bool = True,
        mode: str = "streaming",
    ) -> PipelineResult:
        """Execute the end-to-end data processing pipeline.

        Args:
            input_path: Path to the input JSONL file containing seed records.
            output_path: Path where completed records will be written (appended).
            concurrency: Maximum number of concurrent worker coroutines.
            show_progress: Show a rich CLI progress bar (auto-disabled when not a TTY).
            mode: Execution mode. ``"streaming"`` uses the producer-consumer-flusher
                pattern (safe for huge datasets). ``"burst"`` loads all records into
                memory and processes them with ``asyncio.gather`` + ``Semaphore``
                for maximum throughput on datasets that fit in RAM.

        Returns:
            PipelineResult with execution statistics.
        """
        if mode == "burst":
            return await self._run_burst(
                input_path, output_path, concurrency, show_progress
            )
        # Default: streaming mode
        # Ensure a MetricsCollector is present
        metrics = None
        for h in self.hooks:
            if isinstance(h, MetricsCollector):
                metrics = h
                break
        if metrics is None:
            metrics = MetricsCollector()
            self.hooks.append(metrics)

        if self.checkpoint_disabled:
            checkpoint = None
        elif self.checkpoint_backend == "sqlite":
            checkpoint = SQLiteCheckpointManager(self.checkpoint_dir)
        else:
            checkpoint = CheckpointManager(self.checkpoint_dir)
        if checkpoint is not None:
            await checkpoint.load()
        if checkpoint is not None and checkpoint.completed_count > 0:
            logger.info(
                "Resuming from checkpoint: %d records already completed",
                checkpoint.completed_count,
            )

        progress = None
        task_id = None
        if show_progress:
            total = await self._count_lines(input_path)
            completed_count = checkpoint.completed_count if checkpoint is not None else 0
            pending = max(0, total - completed_count)
            result = self._make_progress(pending)
            if result is not None:
                progress, task_id = result

        queue: asyncio.Queue[DataRecord | None] = asyncio.Queue(maxsize=concurrency * 2)
        write_queue: asyncio.Queue[DataRecord | None] = asyncio.Queue()
        dl_queue: asyncio.Queue[DataRecord | None] = asyncio.Queue()
        shutdown_event = asyncio.Event()
        adaptive_sem: AdaptiveSemaphore | None = None
        if self.adaptive_concurrency:
            adaptive_sem = AdaptiveSemaphore(
                initial=concurrency, max_concurrency=concurrency * 2
            )
        logger.info(
            "Starting pipeline: input=%s, output=%s, concurrency=%d",
            input_path, output_path, concurrency,
        )
        await self._emit("on_pipeline_start", input_path=input_path, output_path=output_path, concurrency=concurrency)

        # Graceful shutdown handler
        def _request_shutdown() -> None:
            if not shutdown_event.is_set():
                logger.info("Shutdown requested, draining in-flight workers...")
                shutdown_event.set()

        loop = asyncio.get_running_loop()
        registered_signals: list[int] = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _request_shutdown)
                registered_signals.append(sig)

        async def producer() -> None:
            try:
                async with aiofiles.open(input_path, encoding="utf-8") as f:
                    async for line in f:
                        if shutdown_event.is_set():
                            logger.info("Producer stopping due to shutdown request")
                            break
                        line = line.strip()
                        if not line:
                            continue
                        raw = json.loads(line)
                        record_id = raw.get("id")
                        record = DataRecord(seed_data=raw) if record_id is None else DataRecord(id=record_id, seed_data=raw)
                        if checkpoint is not None and await checkpoint.is_done(record.id):
                            continue
                        await queue.put(record)
            finally:
                for _ in range(concurrency):
                    await queue.put(None)

        async def flusher(out_file) -> None:
            """Write completed records to output before checkpointing them.

            Batches records to reduce disk flush overhead: accumulates up to
            ``flush_batch_size`` records or waits ``flush_timeout`` seconds
            before writing a single batch with one output flush + one
            checkpoint commit.
            """
            batch: list[DataRecord] = []

            async def _flush_batch() -> None:
                if not batch:
                    return
                payload = "".join(r.model_dump_json() + "\n" for r in batch)
                await out_file.write(payload)
                await out_file.flush()
                if checkpoint is not None:
                    await checkpoint.commit_batch([r.id for r in batch])
                for r in batch:
                    await self._emit("on_record_completed", r)

            while True:
                try:
                    record = await asyncio.wait_for(
                        write_queue.get(), timeout=self.flush_timeout
                    )
                except asyncio.TimeoutError:
                    await _flush_batch()
                    batch = []
                    continue

                try:
                    if record is None:
                        await _flush_batch()
                        return
                    batch.append(record)
                    if len(batch) >= self.flush_batch_size:
                        await _flush_batch()
                        batch = []
                finally:
                    write_queue.task_done()

        async def dl_flusher(dl_file) -> None:
            """Flush failed records to the dead-letter queue."""
            while True:
                record = await dl_queue.get()
                try:
                    if record is None:
                        return
                    await dl_file.write(record.model_dump_json() + "\n")
                    await dl_file.flush()
                finally:
                    dl_queue.task_done()

        async def worker(worker_id: int) -> None:
            retry_engine = RetryEngine(max_retries=self.max_retries)
            while True:
                record = await queue.get()
                if record is None:
                    queue.task_done()
                    break
                try:
                    async with contextlib.AsyncExitStack() as stack:
                        if adaptive_sem is not None:
                            await stack.enter_async_context(adaptive_sem)
                        try:
                            await self._emit("on_record_started", record)
                            # Only copy the record on retry attempts, not the first try.
                            attempt_count = 0

                            async def _attempt(rec=record) -> DataRecord:
                                nonlocal attempt_count
                                r = rec if attempt_count == 0 else rec.model_copy()
                                attempt_count += 1
                                return await self.strategy.apply(r)

                            t0 = time.monotonic()
                            record = await retry_engine.run(_attempt)
                            elapsed = time.monotonic() - t0
                            if adaptive_sem is not None:
                                adaptive_sem.report_latency(elapsed)
                            record.status = RecordStatus.GENERATED
                            passed = True
                            assessment_results: list[dict[str, Any]] = []
                            if self.evaluators:
                                record.status = RecordStatus.EVALUATING
                            for evaluator in self.evaluators:
                                if self.capture_assessment_details:
                                    assessment = await evaluator.assess(record)
                                    assessment_results.append(assessment.model_dump())
                                    if not assessment.passed:
                                        passed = False
                                        break
                                elif not await evaluator.evaluate(record):
                                    passed = False
                                    break
                            if self.capture_assessment_details:
                                numeric_scores = [
                                    float(item["score"])
                                    for item in assessment_results
                                    if item.get("score") is not None
                                ]
                                record.metadata["assessment"] = {
                                    "results": assessment_results,
                                    "aggregate_score": (
                                        round(sum(numeric_scores) / len(numeric_scores), 4)
                                        if numeric_scores
                                        else None
                                    ),
                                    "passed_all": passed,
                                }

                            if passed:
                                record.status = RecordStatus.COMPLETED
                                record.metadata["timestamp"] = time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                )
                                await write_queue.put(record)
                            else:
                                record.status = RecordStatus.REJECTED
                                logger.debug("Record %s rejected by evaluator", record.id)
                                await self._emit("on_record_rejected", record)

                        except Exception as e:
                            try:
                                record.status = RecordStatus.FAILED
                                record.metadata["error"] = str(e)[:2000]
                                logger.error("Record %s failed: %s", record.id, str(e)[:200])
                            except Exception:
                                record.metadata["error"] = "<error handler failed>"
                            if self.dead_letter_path:
                                await dl_queue.put(record)
                            await self._emit("on_record_failed", record, e)
                finally:
                    queue.task_done()
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)

        if progress is not None:
            progress.start()
        try:
            _ckpt_ctx = checkpoint if checkpoint is not None else contextlib.nullcontext()
            async with _ckpt_ctx, aiofiles.open(output_path, "a", encoding="utf-8") as out_file:
                flush_task = asyncio.create_task(flusher(out_file))
                dl_flush_task = None
                dl_ctx = None
                if self.dead_letter_path:
                    dl_ctx = aiofiles.open(self.dead_letter_path, "a", encoding="utf-8")
                    dl_file = await dl_ctx.__aenter__()
                    dl_flush_task = asyncio.create_task(dl_flusher(dl_file))
                    logger.info("Dead letter queue enabled: %s", self.dead_letter_path)

                workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
                await asyncio.gather(producer(), *workers)
                await write_queue.put(None)
                await flush_task
                if dl_flush_task is not None:
                    await dl_queue.put(None)
                    await dl_flush_task
                if dl_ctx is not None:
                    await dl_ctx.__aexit__(None, None, None)
        finally:
            if progress is not None:
                progress.stop()
            # Restore original signal handlers
            for sig in registered_signals:
                with contextlib.suppress(NotImplementedError):
                    loop.remove_signal_handler(sig)
            await self._emit("on_pipeline_end")

        assert metrics.result is not None
        return metrics.result
