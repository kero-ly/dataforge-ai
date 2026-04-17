# tests/test_cli_subcommands.py
"""Tests for CLI subcommands: validate, status, inspect, version."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from dataforge.schema import DataRecord, RecordStatus


def _make_config(tmpdir: str) -> str:
    """Create a minimal valid YAML config and return its path."""
    config = {
        "name": "test-pipeline",
        "source": {"type": "jsonl", "path": f"{tmpdir}/seeds.jsonl"},
        "pipeline": [
            {
                "step": "generate",
                "strategy": "evol-instruct",
                "llm": {"provider": "vllm", "model": "test", "base_url": "http://localhost:8000/v1"},
            }
        ],
        "sink": {"path": f"{tmpdir}/output.jsonl"},
    }
    config_path = f"{tmpdir}/config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_validate_valid_config(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _make_config(tmpdir)

        from dataforge.cli import _validate

        _validate(config_path)

        output = capsys.readouterr().out
        assert "Config is valid" in output
        assert "test-pipeline" in output
        assert "generate" in output


def test_validate_invalid_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = f"{tmpdir}/bad.yaml"
        with open(config_path, "w") as f:
            f.write("name: missing-required-fields\n")

        from dataforge.cli import _validate

        with pytest.raises(SystemExit) as exc_info:
            _validate(config_path)
        assert exc_info.value.code == 1


def test_status_with_checkpoint(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        wal = ckpt_dir / "checkpoint.jsonl"
        with open(wal, "w") as f:
            f.write(json.dumps({"id": "rec-001"}) + "\n")
            f.write(json.dumps({"id": "rec-002"}) + "\n")
            f.write(json.dumps({"id": "rec-003"}) + "\n")

        from dataforge.cli import _status

        _status(str(ckpt_dir))

        output = capsys.readouterr().out
        assert "Completed records: 3" in output


def test_status_no_checkpoint(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        from dataforge.cli import _status

        _status(tmpdir)

        output = capsys.readouterr().out
        assert "No checkpoint found" in output


def test_status_with_malformed_lines(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        wal = ckpt_dir / "checkpoint.jsonl"
        with open(wal, "w") as f:
            f.write(json.dumps({"id": "rec-001"}) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps({"id": "rec-002"}) + "\n")

        from dataforge.cli import _status

        _status(str(ckpt_dir))

        output = capsys.readouterr().out
        assert "Completed records: 2" in output
        assert "Malformed lines:   1" in output


def test_inspect_output_file(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.jsonl"
        records = [
            DataRecord(seed_data={"q": "a"}, synthetic_data={"r": "b"}, status=RecordStatus.COMPLETED, score=4.5),
            DataRecord(seed_data={"q": "c"}, synthetic_data={"r": "d"}, status=RecordStatus.COMPLETED, score=3.0),
        ]
        with open(output_path, "w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")

        from dataforge.cli import _inspect

        _inspect(str(output_path))

        output = capsys.readouterr().out
        assert "Total records: 2" in output
        assert "COMPLETED: 2" in output
        assert "Avg score: 3.75" in output


def test_inspect_missing_file():
    from dataforge.cli import _inspect

    with pytest.raises(SystemExit) as exc_info:
        _inspect("/nonexistent/path/output.jsonl")
    assert exc_info.value.code == 1


def test_version(capsys):
    from dataforge.cli import _version

    _version()
    output = capsys.readouterr().out
    assert "dataforge" in output
