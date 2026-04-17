# src/dataforge/cli.py
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def _setup_logging(level: str, log_file: str | None = None) -> None:
    """Configure root logging for the dataforge package."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []

    try:
        from rich.logging import RichHandler

        handlers.append(RichHandler(level=numeric_level, rich_tracebacks=True, show_path=False))
    except ImportError:
        handlers.append(logging.StreamHandler())

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(name)s - %(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding existing env."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _autoload_env(config_path: str | None = None) -> None:
    """Auto-load .env from cwd and, when available, the config file directory."""
    candidates: list[Path] = [Path.cwd() / ".env"]
    if config_path is not None:
        config_env = Path(config_path).resolve().parent / ".env"
        if config_env not in candidates:
            candidates.append(config_env)

    for env_path in candidates:
        _load_env_file(env_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataforge",
        description="DataForge — high-concurrency LLM data synthesis pipeline",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: WARNING)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Shortcut for --log-level INFO",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write logs to this file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = subparsers.add_parser(
        "run", help="Run a synthesis pipeline from a YAML config file"
    )
    run_parser.add_argument(
        "config", metavar="CONFIG", help="Path to YAML config file"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without running the pipeline",
    )

    assess_parser = subparsers.add_parser(
        "assess", help="Run a dataset assessment from a YAML config file"
    )
    assess_parser.add_argument(
        "config", metavar="CONFIG", help="Path to assessment YAML config file"
    )
    assess_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate assessment config and exit without running",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run a benchmark from a YAML config file"
    )
    benchmark_parser.add_argument(
        "config", metavar="CONFIG", help="Path to benchmark YAML config file"
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate benchmark config and exit without running",
    )

    # --- validate ---
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a YAML config file without running anything"
    )
    validate_parser.add_argument(
        "config", metavar="CONFIG", help="Path to YAML config file"
    )

    # --- status ---
    status_parser = subparsers.add_parser(
        "status", help="Show checkpoint status for a pipeline run"
    )
    status_parser.add_argument(
        "checkpoint_dir", metavar="CHECKPOINT_DIR",
        help="Path to the checkpoint directory (e.g. ./.dataforge_runs)",
    )

    # --- inspect ---
    inspect_parser = subparsers.add_parser(
        "inspect", help="Show statistics for an output JSONL file"
    )
    inspect_parser.add_argument(
        "output_file", metavar="OUTPUT_FILE", help="Path to output JSONL file"
    )

    # --- version ---
    subparsers.add_parser("version", help="Print the DataForge version")

    # --- shard ---
    shard_parser = subparsers.add_parser(
        "shard", help="Distributed file-based sharding commands"
    )
    shard_sub = shard_parser.add_subparsers(dest="shard_command", required=True)

    # shard split
    split_parser = shard_sub.add_parser("split", help="Split input file into N shards")
    split_parser.add_argument("--input", required=True, help="Path to input file")
    split_parser.add_argument(
        "--num-shards", type=int, required=True, help="Number of shards"
    )
    split_parser.add_argument(
        "--output-dir", required=True, help="Directory for shard output files"
    )
    split_parser.add_argument(
        "--format", default="jsonl", choices=["jsonl", "csv", "parquet"],
        help="Input file format (default: jsonl)",
    )

    # shard config
    config_gen_parser = shard_sub.add_parser(
        "config", help="Generate per-shard YAML configs from a template"
    )
    config_gen_parser.add_argument(
        "--template", required=True, help="Path to template YAML config"
    )
    config_gen_parser.add_argument(
        "--shard-dir", required=True, help="Directory containing shard files"
    )
    config_gen_parser.add_argument(
        "--api-keys", nargs="*", default=None,
        help="API keys to distribute across shards",
    )

    # shard run
    shard_run_parser = shard_sub.add_parser(
        "run", help="Run all shard configs as parallel pipelines"
    )
    shard_run_parser.add_argument(
        "--shard-dir", required=True, help="Directory containing shard configs"
    )
    shard_run_parser.add_argument(
        "--sequential", action="store_true",
        help="Run shards sequentially instead of in parallel",
    )

    # shard merge
    merge_parser = shard_sub.add_parser("merge", help="Merge shard outputs into one file")
    merge_parser.add_argument(
        "--shard-dir", required=True, help="Directory containing shard output files"
    )
    merge_parser.add_argument(
        "--output", required=True, help="Path to final merged output file"
    )
    merge_parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip deduplication by record ID",
    )

    # shard status
    shard_status_parser = shard_sub.add_parser(
        "status", help="Show progress of each shard"
    )
    shard_status_parser.add_argument(
        "--shard-dir", required=True, help="Directory containing shard checkpoints"
    )

    # --- coordinator ---
    coord_parser = subparsers.add_parser(
        "coordinator", help="Run as distributed coordinator (requires Redis)"
    )
    coord_parser.add_argument(
        "config", metavar="CONFIG", help="Path to YAML config file"
    )
    coord_parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Expected number of workers (for progress tracking)",
    )

    # --- worker ---
    worker_parser = subparsers.add_parser(
        "worker", help="Run as distributed worker (requires Redis)"
    )
    worker_parser.add_argument(
        "config", metavar="CONFIG", help="Path to YAML config file"
    )
    worker_parser.add_argument(
        "--worker-id", default="worker-0",
        help="Unique identifier for this worker",
    )
    worker_parser.add_argument(
        "--concurrency", type=int, default=None,
        help="Override concurrency for this worker",
    )

    args = parser.parse_args()

    config_path = getattr(args, "config", None)
    _autoload_env(config_path)

    log_level = "INFO" if args.verbose else args.log_level
    _setup_logging(log_level, args.log_file)

    if args.command == "run":
        _run(args.config, dry_run=args.dry_run)
    elif args.command == "assess":
        _assess(args.config, dry_run=args.dry_run)
    elif args.command == "benchmark":
        _benchmark(args.config, dry_run=args.dry_run)
    elif args.command == "validate":
        _validate(args.config)
    elif args.command == "status":
        _status(args.checkpoint_dir)
    elif args.command == "inspect":
        _inspect(args.output_file)
    elif args.command == "version":
        _version()
    elif args.command == "shard":
        _shard(args)
    elif args.command == "coordinator":
        _coordinator(args)
    elif args.command == "worker":
        _worker(args)


def _run(config_path: str, dry_run: bool = False) -> None:
    logger = logging.getLogger(__name__)

    from dataforge.config import build_pipeline, load_config

    _autoload_env(config_path)

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error("Failed to parse config '%s': %s", config_path, e)
        sys.exit(1)

    if dry_run:
        logger.info("Config '%s' is valid.", config.name)
        return

    try:
        pipeline, input_path, output_path, concurrency = build_pipeline(config)
    except Exception as e:
        logger.error("Invalid pipeline config: %s", e)
        sys.exit(1)

    result = asyncio.run(
        pipeline.run(
            input_path=input_path,
            output_path=output_path,
            concurrency=concurrency,
            mode=config.mode,
        )
    )

    print(result.summary())


def _validate(config_path: str) -> None:
    logger = logging.getLogger(__name__)

    from dataforge.config import load_config

    _autoload_env(config_path)

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error("Invalid config '%s': %s", config_path, e)
        sys.exit(1)

    logger.info("Config '%s' is valid. Pipeline: %s", config.name, config_path)
    print(f"Config is valid: {config.name}")
    print(f"  Source: {config.source.type} @ {config.source.path}")
    print(f"  Sink:   {config.sink.path}")
    print(f"  Steps:  {len(config.pipeline)}")
    for i, step in enumerate(config.pipeline, 1):
        if step.step == "generate":
            print(f"    {i}. generate ({step.strategy})")
        elif step.step == "evaluate":
            print(f"    {i}. evaluate ({step.evaluator})")


def _assess(config_path: str, dry_run: bool = False) -> None:
    logger = logging.getLogger(__name__)

    import dataforge.assessment.suite  # noqa: F401

    from dataforge.config import build_assessment_runner, load_assessment_config
    from dataforge.registry import get_assessment_suite

    _autoload_env(config_path)
    try:
        config = load_assessment_config(config_path)
        get_assessment_suite(config.suite.name)
    except Exception as e:
        logger.error("Invalid assessment config '%s': %s", config_path, e)
        sys.exit(1)

    if dry_run:
        print(f"Assessment config is valid: {config.name}")
        print(f"  Source: {config.source.path} ({config.source.format})")
        print(f"  Suite:  {config.suite.name}")
        return

    output_dir, summary = asyncio.run(build_assessment_runner(config).run())
    print(f"Assessment complete: {output_dir}")
    print(f"  Suite: {summary.suite}")
    print(f"  Overall quality score: {summary.overall_quality_score}")
    print(f"  Sampled records: {summary.sampled_records}")


def _benchmark(config_path: str, dry_run: bool = False) -> None:
    logger = logging.getLogger(__name__)

    import dataforge.benchmark.tasks  # noqa: F401

    from dataforge.config import build_benchmark_runner, load_benchmark_config
    from dataforge.registry import get_benchmark

    _autoload_env(config_path)
    try:
        config = load_benchmark_config(config_path)
        for task in config.tasks:
            get_benchmark(task.name)
    except Exception as e:
        logger.error("Invalid benchmark config '%s': %s", config_path, e)
        sys.exit(1)

    if dry_run:
        print(f"Benchmark config is valid: {config.name}")
        print(f"  Candidate: {config.candidate.model}")
        print(f"  Tasks: {', '.join(task.name for task in config.tasks)}")
        return

    output_dir, summary = asyncio.run(build_benchmark_runner(config).run())
    print(f"Benchmark complete: {output_dir}")
    print(f"  Candidate: {summary.candidate_name}")
    print(f"  Overall score: {summary.overall_score}")


def _status(checkpoint_dir: str) -> None:
    from pathlib import Path

    wal_path = Path(checkpoint_dir) / "checkpoint.jsonl"
    if not wal_path.exists():
        print(f"No checkpoint found at {checkpoint_dir}")
        return

    count = 0
    errors = 0
    with open(wal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except (json.JSONDecodeError, ValueError):
                errors += 1

    print(f"Checkpoint: {wal_path}")
    print(f"  Completed records: {count}")
    if errors:
        print(f"  Malformed lines:   {errors}")


def _inspect(output_file: str) -> None:
    from pathlib import Path

    from dataforge.schema import DataRecord, RecordStatus

    path = Path(output_file)
    if not path.exists():
        print(f"File not found: {output_file}")
        sys.exit(1)

    total = 0
    status_counts: dict[str, int] = {}
    scores: list[float] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = DataRecord.model_validate_json(line)
                status_name = record.status.value if isinstance(record.status, RecordStatus) else str(record.status)
                status_counts[status_name] = status_counts.get(status_name, 0) + 1
                if record.score is not None:
                    scores.append(record.score)
            except Exception:
                status_counts["PARSE_ERROR"] = status_counts.get("PARSE_ERROR", 0) + 1

    print(f"Output: {output_file}")
    print(f"  Total records: {total}")
    for status, cnt in sorted(status_counts.items()):
        print(f"  {status}: {cnt}")
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Avg score: {avg:.2f} (n={len(scores)})")


def _version() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("dataforge")
    except PackageNotFoundError:
        v = "dev (not installed)"
    print(f"dataforge {v}")


def _shard(args: argparse.Namespace) -> None:
    from dataforge.distributed.shard import (
        generate_shard_configs,
        merge_outputs,
        run_shards,
        shard_status,
        split_input,
    )

    if args.shard_command == "split":
        paths = split_input(
            input_path=args.input,
            num_shards=args.num_shards,
            output_dir=args.output_dir,
            format=args.format,
        )
        print(f"Split into {len(paths)} shards:")
        for p in paths:
            print(f"  {p}")

    elif args.shard_command == "config":
        paths = generate_shard_configs(
            template_config_path=args.template,
            shard_dir=args.shard_dir,
            api_keys=args.api_keys,
        )
        print(f"Generated {len(paths)} shard configs:")
        for p in paths:
            print(f"  {p}")

    elif args.shard_command == "run":
        codes = asyncio.run(
            run_shards(
                shard_dir=args.shard_dir,
                parallel=not args.sequential,
            )
        )
        failed = sum(1 for c in codes if c != 0)
        print(f"Completed: {len(codes) - failed}/{len(codes)} shards succeeded")
        if failed:
            sys.exit(1)

    elif args.shard_command == "merge":
        total = merge_outputs(
            shard_dir=args.shard_dir,
            output_path=args.output,
            dedup=not args.no_dedup,
        )
        print(f"Merged {total} records → {args.output}")

    elif args.shard_command == "status":
        statuses = shard_status(args.shard_dir)
        if not statuses:
            print("No shard checkpoints found.")
            return
        total = 0
        for s in statuses:
            print(f"  Shard {s['shard_id']}: {s['completed_count']} completed")
            total += s["completed_count"]
        print(f"  Total: {total} completed across {len(statuses)} shards")


def _coordinator(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    from dataforge.config import load_config
    from dataforge.distributed.coordinator import Coordinator

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Failed to parse config '%s': %s", args.config, e)
        sys.exit(1)

    dist = config.distributed
    if dist is None:
        logger.error("Config must have a 'distributed' section for coordinator mode")
        sys.exit(1)

    coordinator = Coordinator(
        redis_url=dist.redis_url,
        run_id=config.name,
        queue_name=dist.queue_name,
        result_channel=dist.result_channel,
    )

    result = asyncio.run(
        coordinator.run(
            input_path=config.source.path,
            output_path=config.sink.path,
            num_workers=args.num_workers,
        )
    )

    print(result.summary())


def _worker(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)

    from dataforge.config import build_pipeline, load_config
    from dataforge.distributed.worker import DistributedWorker

    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error("Failed to parse config '%s': %s", args.config, e)
        sys.exit(1)

    dist = config.distributed
    if dist is None:
        logger.error("Config must have a 'distributed' section for worker mode")
        sys.exit(1)

    try:
        pipeline, _, _, concurrency = build_pipeline(config)
    except Exception as e:
        logger.error("Invalid pipeline config: %s", e)
        sys.exit(1)

    if args.concurrency is not None:
        concurrency = args.concurrency

    worker = DistributedWorker(
        pipeline=pipeline,
        redis_url=dist.redis_url,
        run_id=config.name,
        queue_name=dist.queue_name,
        result_channel=dist.result_channel,
        concurrency=concurrency,
        worker_id=args.worker_id,
    )

    result = asyncio.run(worker.run())
    print(result.summary())
