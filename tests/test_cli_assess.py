import json
import tempfile
from pathlib import Path


def test_assess_dry_run(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.jsonl"
        data_path.write_text(json.dumps({"instruction": "q", "response": "a"}) + "\n")
        config_path = Path(tmpdir) / "assessment.yaml"
        config_path.write_text(
            "\n".join(
                [
                    'kind: "assessment"',
                    'name: "demo"',
                    "source:",
                    f'  path: "{data_path}"',
                    '  format: "auto"',
                    "suite:",
                    '  name: "sft_readiness_v1"',
                    "output:",
                    f'  dir: "{tmpdir}"',
                ]
            ),
            encoding="utf-8",
        )
        from dataforge.cli import _assess

        _assess(str(config_path), dry_run=True)

        assert "Assessment config is valid" in capsys.readouterr().out
