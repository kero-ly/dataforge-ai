# tests/test_cli.py
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


def _write_seeds(path: Path, n: int = 2) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"instruction": f"question {i}"}) + "\n")


def test_dry_run_prints_valid(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "name": "dry-run-test",
            "source": {"type": "jsonl", "path": f"{tmpdir}/seeds.jsonl"},
            "pipeline": [
                {
                    "step": "generate",
                    "strategy": "evol-instruct",
                    "depth": 1,
                    "llm": {"provider": "vllm", "model": "test-model"},
                }
            ],
            "sink": {"path": f"{tmpdir}/output.jsonl"},
        }
        config_path = f"{tmpdir}/config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        import logging

        from dataforge.cli import _run
        with caplog.at_level(logging.INFO):
            _run(config_path, dry_run=True)

        assert "valid" in caplog.text.lower()
        assert "dry-run-test" in caplog.text


def test_run_end_to_end_with_fake_llm():
    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = Path(tmpdir) / "seeds.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        _write_seeds(seeds_path, n=2)

        config = {
            "name": "e2e-test",
            "source": {"type": "jsonl", "path": str(seeds_path)},
            "pipeline": [
                {
                    "step": "generate",
                    "strategy": "evol-instruct",
                    "depth": 1,
                    "llm": {
                        "provider": "vllm",
                        "model": "fake-model",
                        "concurrency": 2,
                    },
                }
            ],
            "sink": {
                "path": str(output_path),
                "checkpoint_dir": f"{tmpdir}/ckpt",
            },
        }
        config_path = f"{tmpdir}/config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from dataforge.clients.vllm_client import vLLMClient

        async def fake_generate(self, prompt, **kwargs):
            return "evolved instruction"

        with patch.object(vLLMClient, "generate", fake_generate):
            from dataforge.cli import _run
            _run(config_path, dry_run=False)

        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 2


def test_invalid_config_exits_with_error(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = f"{tmpdir}/bad.yaml"
        with open(config_path, "w") as f:
            f.write("name: missing-required-fields\n")

        import logging

        from dataforge.cli import _run
        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
            _run(config_path, dry_run=False)
        assert exc_info.value.code == 1
        assert "error" in caplog.text.lower() or "failed" in caplog.text.lower()


def test_run_autoloads_env_from_config_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        seeds_path = Path(tmpdir) / "seeds.jsonl"
        output_path = Path(tmpdir) / "output.jsonl"
        config_path = Path(tmpdir) / "config.yaml"
        env_path = Path(tmpdir) / ".env"
        _write_seeds(seeds_path, n=1)

        config = {
            "name": "autoload-env-test",
            "source": {"type": "jsonl", "path": str(seeds_path)},
            "pipeline": [
                {
                    "step": "generate",
                    "strategy": "evol-instruct",
                    "depth": 1,
                    "llm": {
                        "provider": "bailian",
                        "model": "qwen-plus",
                        "concurrency": 1,
                    },
                }
            ],
            "sink": {
                "path": str(output_path),
                "checkpoint_dir": f"{tmpdir}/ckpt",
            },
        }

        config_path.write_text(yaml.dump(config), encoding="utf-8")
        env_path.write_text("DASHSCOPE_API_KEY=from-dotenv\n", encoding="utf-8")

        from dataforge.cli import _run
        from dataforge.clients.bailian_client import BailianClient

        async def fake_generate(self, prompt, **kwargs):
            return "evolved instruction"

        with patch.dict("os.environ", {}, clear=True), patch.object(
            BailianClient, "generate", fake_generate
        ):
            _run(str(config_path), dry_run=False)

        assert output_path.exists()


def test_run_autoload_does_not_override_existing_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        env_path = Path(tmpdir) / ".env"

        config = {
            "name": "autoload-env-priority-test",
            "source": {"type": "jsonl", "path": f"{tmpdir}/seeds.jsonl"},
            "pipeline": [
                {
                    "step": "generate",
                    "strategy": "evol-instruct",
                    "depth": 1,
                    "llm": {"provider": "bailian", "model": "qwen-plus"},
                }
            ],
            "sink": {"path": f"{tmpdir}/output.jsonl"},
        }

        config_path.write_text(yaml.dump(config), encoding="utf-8")
        env_path.write_text("DASHSCOPE_API_KEY=from-dotenv\n", encoding="utf-8")

        from dataforge.cli import _autoload_env

        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "from-shell"}, clear=True):
            _autoload_env(str(config_path))
            assert os.environ["DASHSCOPE_API_KEY"] == "from-shell"
