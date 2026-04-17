from __future__ import annotations

from experiments.exp8_framework_baselines import run_datajuicer, run_distilabel


def test_distilabel_cli_accepts_api_metadata(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_distilabel.py",
            "--dataset",
            "seeds.jsonl",
            "--backend",
            "deepseek",
            "--provider",
            "deepseek",
            "--rpm-limit",
            "300",
            "--tpm-limit",
            "50000",
        ],
    )
    args = run_distilabel.parse_args()
    assert args.backend == "deepseek"
    assert args.provider == "deepseek"
    assert args.rpm_limit == 300
    assert args.tpm_limit == 50000


def test_datajuicer_cli_accepts_api_metadata(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datajuicer.py",
            "--dataset",
            "seeds.jsonl",
            "--backend",
            "deepseek",
            "--provider",
            "deepseek",
            "--rpm-limit",
            "300",
            "--tpm-limit",
            "50000",
        ],
    )
    args = run_datajuicer.parse_args()
    assert args.backend == "deepseek"
    assert args.provider == "deepseek"
    assert args.rpm_limit == 300
    assert args.tpm_limit == 50000
