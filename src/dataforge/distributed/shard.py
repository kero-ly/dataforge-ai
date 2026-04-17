# src/dataforge/distributed/shard.py
"""File-based sharding for horizontal scaling."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def split_input(
    input_path: str,
    num_shards: int,
    output_dir: str,
    *,
    format: str = "jsonl",
) -> list[str]:
    """Split an input file into N shards using round-robin distribution."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        lines = _read_jsonl_lines(input_path)
    else:
        from dataforge.io import read_records
        raw_records = read_records(input_path, format)
        lines = [json.dumps(r, ensure_ascii=False) for r in raw_records]

    shard_lines: list[list[str]] = [[] for _ in range(num_shards)]
    for i, line in enumerate(lines):
        shard_lines[i % num_shards].append(line)

    shard_paths: list[str] = []
    for shard_id in range(num_shards):
        shard_path = out / f"shard_{shard_id}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for line in shard_lines[shard_id]:
                f.write(line + "\n")
        shard_paths.append(str(shard_path))
        logger.info("Shard %d: %d records -> %s", shard_id, len(shard_lines[shard_id]), shard_path)

    return shard_paths


def _read_jsonl_lines(path: str) -> list[str]:
    lines: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def generate_shard_configs(
    template_config_path: str,
    shard_dir: str,
    api_keys: list[str] | None = None,
) -> list[str]:
    """Generate per-shard YAML configs from a template."""
    with open(template_config_path, encoding="utf-8") as f:
        template: dict[str, Any] = yaml.safe_load(f)

    shard_path = Path(shard_dir)
    shard_files = sorted(shard_path.glob("shard_*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    config_paths: list[str] = []
    for shard_id, shard_file in enumerate(shard_files):
        config = _deep_copy_dict(template)
        config["source"] = {"type": "jsonl", "path": str(shard_file)}

        sink = config.setdefault("sink", {})
        original_output = sink.get("path", "output.jsonl")
        stem = Path(original_output).stem
        suffix = Path(original_output).suffix or ".jsonl"
        sink["path"] = str(shard_path / f"{stem}_shard_{shard_id}{suffix}")
        sink["checkpoint_dir"] = str(shard_path / f"checkpoint_shard_{shard_id}")

        if api_keys:
            key = api_keys[shard_id % len(api_keys)]
            _set_api_key(config, key)

        config["name"] = f"{config.get('name', 'pipeline')}_shard_{shard_id}"

        config_path = shard_path / f"config_shard_{shard_id}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        config_paths.append(str(config_path))

    return config_paths


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = [_deep_copy_dict(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


def _set_api_key(config: dict[str, Any], api_key: str) -> None:
    for step in config.get("pipeline", []):
        if "llm" in step and isinstance(step["llm"], dict):
            step["llm"]["api_key"] = api_key


def merge_outputs(
    shard_dir: str,
    output_path: str,
    *,
    dedup: bool = True,
) -> int:
    """Merge all shard output files into a single output file."""
    shard_path = Path(shard_dir)
    output_files = sorted(shard_path.glob("*_shard_*.jsonl"))
    output_files = [f for f in output_files if not f.name.startswith("shard_")]

    seen_ids: set[str] = set()
    total = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for output_file in output_files:
            with open(output_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if dedup:
                        try:
                            record = json.loads(line)
                            record_id = record.get("id", "")
                            if record_id in seen_ids:
                                continue
                            seen_ids.add(record_id)
                        except json.JSONDecodeError:
                            continue
                    out.write(line + "\n")
                    total += 1

    logger.info("Merged %d records from %d shards -> %s", total, len(output_files), output_path)
    return total


def shard_status(shard_dir: str) -> list[dict[str, Any]]:
    """Get the status of each shard's checkpoint."""
    shard_path = Path(shard_dir)
    checkpoint_dirs = sorted(shard_path.glob("checkpoint_shard_*"))

    statuses: list[dict[str, Any]] = []
    for ckpt_dir in checkpoint_dirs:
        shard_id_str = ckpt_dir.name.split("_")[-1]
        wal = ckpt_dir / "checkpoint.jsonl"
        count = 0
        if wal.exists():
            with open(wal, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        statuses.append({
            "shard_id": int(shard_id_str),
            "checkpoint_dir": str(ckpt_dir),
            "completed_count": count,
        })

    return statuses


async def run_shards(
    shard_dir: str,
    *,
    parallel: bool = True,
    extra_args: list[str] | None = None,
) -> list[int]:
    """Run all shard configs as subprocess calls to dataforge run."""
    shard_path = Path(shard_dir)
    config_files = sorted(shard_path.glob("config_shard_*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No shard configs found in {shard_dir}")

    async def _run_one(config_path: Path) -> int:
        cmd = [sys.executable, "-m", "dataforge", "run", str(config_path)]
        if extra_args:
            cmd.extend(extra_args)
        logger.info("Starting shard: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        rc = proc.returncode or 0
        if rc != 0:
            logger.error("Shard %s failed (rc=%d): %s", config_path.name, rc, stderr.decode()[:500])
        else:
            logger.info("Shard %s completed successfully", config_path.name)
        return rc

    if parallel:
        results = await asyncio.gather(*[_run_one(cfg) for cfg in config_files])
        return list(results)
    else:
        results = []
        for cfg in config_files:
            rc = await _run_one(cfg)
            results.append(rc)
        return results
