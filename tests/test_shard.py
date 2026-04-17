# tests/test_shard.py
"""Tests for file-based sharding (Phase 1 distributed mode)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from dataforge.distributed.shard import (
    merge_outputs,
    shard_status,
    split_input,
    generate_shard_configs,
)


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> str:
    """Create a sample JSONL file with 10 records."""
    path = tmp_path / "seeds.jsonl"
    records = [{"id": f"rec_{i}", "instruction": f"Question {i}"} for i in range(10)]
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture
def sample_config(tmp_path: Path, sample_jsonl: str) -> str:
    """Create a sample YAML config file."""
    config = {
        "name": "test-pipeline",
        "source": {"type": "jsonl", "path": sample_jsonl},
        "pipeline": [
            {
                "step": "generate",
                "strategy": "evol-instruct",
                "llm": {
                    "provider": "vllm",
                    "model": "test-model",
                    "base_url": "http://localhost:8000/v1",
                },
            }
        ],
        "sink": {"path": str(tmp_path / "output.jsonl")},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


class TestSplitInput:
    def test_split_basic(self, sample_jsonl: str, tmp_path: Path) -> None:
        shard_dir = str(tmp_path / "shards")
        paths = split_input(sample_jsonl, num_shards=3, output_dir=shard_dir)

        assert len(paths) == 3
        # Check all files exist and total records == 10
        total = 0
        for p in paths:
            assert os.path.exists(p)
            with open(p) as f:
                lines = [l for l in f if l.strip()]
                total += len(lines)
        assert total == 10

    def test_split_round_robin(self, sample_jsonl: str, tmp_path: Path) -> None:
        shard_dir = str(tmp_path / "shards")
        paths = split_input(sample_jsonl, num_shards=3, output_dir=shard_dir)

        # 10 records / 3 shards = 4, 3, 3
        counts = []
        for p in paths:
            with open(p) as f:
                counts.append(sum(1 for l in f if l.strip()))
        assert sorted(counts) == [3, 3, 4]

    def test_split_single_shard(self, sample_jsonl: str, tmp_path: Path) -> None:
        shard_dir = str(tmp_path / "shards")
        paths = split_input(sample_jsonl, num_shards=1, output_dir=shard_dir)

        assert len(paths) == 1
        with open(paths[0]) as f:
            assert sum(1 for l in f if l.strip()) == 10

    def test_split_invalid_num_shards(self, sample_jsonl: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="num_shards must be >= 1"):
            split_input(sample_jsonl, num_shards=0, output_dir=str(tmp_path))

    def test_split_preserves_content(self, sample_jsonl: str, tmp_path: Path) -> None:
        shard_dir = str(tmp_path / "shards")
        paths = split_input(sample_jsonl, num_shards=2, output_dir=shard_dir)

        all_ids = set()
        for p in paths:
            with open(p) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        all_ids.add(record["id"])

        expected_ids = {f"rec_{i}" for i in range(10)}
        assert all_ids == expected_ids


class TestGenerateShardConfigs:
    def test_generate_configs(self, sample_config: str, tmp_path: Path) -> None:
        shard_dir = str(tmp_path / "shards")
        # First split input
        with open(sample_config) as f:
            cfg = yaml.safe_load(f)
        split_input(cfg["source"]["path"], num_shards=2, output_dir=shard_dir)

        config_paths = generate_shard_configs(sample_config, shard_dir)
        assert len(config_paths) == 2

        for i, cp in enumerate(config_paths):
            with open(cp) as f:
                shard_cfg = yaml.safe_load(f)
            assert shard_cfg["source"]["type"] == "jsonl"
            assert f"shard_{i}" in shard_cfg["source"]["path"]
            assert f"shard_{i}" in shard_cfg["sink"]["path"]
            assert f"shard_{i}" in shard_cfg["sink"]["checkpoint_dir"]
            assert f"shard_{i}" in shard_cfg["name"]

    def test_generate_configs_with_api_keys(self, sample_config: str, tmp_path: Path) -> None:
        shard_dir = str(tmp_path / "shards")
        with open(sample_config) as f:
            cfg = yaml.safe_load(f)
        split_input(cfg["source"]["path"], num_shards=3, output_dir=shard_dir)

        api_keys = ["sk-key1", "sk-key2"]
        config_paths = generate_shard_configs(sample_config, shard_dir, api_keys=api_keys)

        for i, cp in enumerate(config_paths):
            with open(cp) as f:
                shard_cfg = yaml.safe_load(f)
            expected_key = api_keys[i % len(api_keys)]
            assert shard_cfg["pipeline"][0]["llm"]["api_key"] == expected_key


class TestMergeOutputs:
    def test_merge_basic(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()

        # Create shard output files
        for shard_id in range(2):
            out_path = shard_dir / f"output_shard_{shard_id}.jsonl"
            records = [
                {"id": f"shard{shard_id}_rec_{i}", "status": "COMPLETED"}
                for i in range(3)
            ]
            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        output_path = str(tmp_path / "merged.jsonl")
        total = merge_outputs(str(shard_dir), output_path)

        assert total == 6
        with open(output_path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 6

    def test_merge_dedup(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()

        # Create overlapping records
        for shard_id in range(2):
            out_path = shard_dir / f"output_shard_{shard_id}.jsonl"
            records = [
                {"id": "shared_id", "status": "COMPLETED", "shard": shard_id},
                {"id": f"unique_{shard_id}", "status": "COMPLETED"},
            ]
            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        output_path = str(tmp_path / "merged.jsonl")
        total = merge_outputs(str(shard_dir), output_path, dedup=True)

        assert total == 3  # 1 shared + 2 unique

    def test_merge_no_dedup(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()

        for shard_id in range(2):
            out_path = shard_dir / f"output_shard_{shard_id}.jsonl"
            with open(out_path, "w") as f:
                f.write(json.dumps({"id": "same_id"}) + "\n")

        output_path = str(tmp_path / "merged.jsonl")
        total = merge_outputs(str(shard_dir), output_path, dedup=False)

        assert total == 2  # No dedup, both kept


class TestShardStatus:
    def test_status_no_checkpoints(self, tmp_path: Path) -> None:
        statuses = shard_status(str(tmp_path))
        assert statuses == []

    def test_status_with_checkpoints(self, tmp_path: Path) -> None:
        # Create checkpoint directories
        for shard_id in range(2):
            ckpt_dir = tmp_path / f"checkpoint_shard_{shard_id}"
            ckpt_dir.mkdir()
            wal = ckpt_dir / "checkpoint.jsonl"
            with open(wal, "w") as f:
                for i in range(shard_id + 3):
                    f.write(json.dumps({"id": f"rec_{i}"}) + "\n")

        statuses = shard_status(str(tmp_path))
        assert len(statuses) == 2
        assert statuses[0]["completed_count"] == 3
        assert statuses[1]["completed_count"] == 4
