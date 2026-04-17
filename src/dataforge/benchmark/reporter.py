from __future__ import annotations

import datetime
import html
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

from dataforge.benchmark.schema import BenchmarkCaseResult, BenchmarkRunSummary


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _timestamp_slug() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_benchmark_report(
    *,
    run_name: str,
    benchmark_name: str,
    candidate_name: str,
    summary: BenchmarkRunSummary,
    task_results: dict[str, list[BenchmarkCaseResult]],
    output_dir: str | Path,
    output_formats: list[str],
    config_snapshot: dict[str, Any],
) -> Path:
    root = Path(output_dir) / run_name / _timestamp_slug()
    task_dir = root / "task_results"
    task_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": run_name,
        "benchmark": benchmark_name,
        "candidate_name": candidate_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
        },
        "git": {"commit": _git_commit(Path(__file__).resolve().parents[3])},
        "config": config_snapshot,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    for task_name, results in task_results.items():
        with open(task_dir / f"{task_name}.jsonl", "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(result.model_dump_json() + "\n")

    if "md" in output_formats:
        (root / "report.md").write_text(_render_markdown(summary), encoding="utf-8")
    if "html" in output_formats:
        (root / "report.html").write_text(_render_html(summary), encoding="utf-8")
    return root


def _render_markdown(summary: BenchmarkRunSummary) -> str:
    lines = [
        f"# Benchmark Report: {summary.benchmark}",
        "",
        f"- Candidate: {summary.candidate_name}",
        f"- Overall score: {summary.overall_score}",
        "",
        "## Tasks",
        "",
        "| Task | Score | Success Rate | Errors |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task in summary.task_summaries:
        lines.append(
            f"| {task.task} | {task.overall_score} | {task.success_rate:.2%} | {task.num_errors} |"
        )
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in summary.warnings])
    return "\n".join(lines) + "\n"


def _render_html(summary: BenchmarkRunSummary) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(task.task)}</td>"
        f"<td>{task.overall_score}</td>"
        f"<td>{task.success_rate:.2%}</td>"
        f"<td>{task.num_errors}</td>"
        "</tr>"
        for task in summary.task_summaries
    )
    warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in summary.warnings)
    return (
        "<html><body>"
        f"<h1>Benchmark Report: {html.escape(summary.benchmark)}</h1>"
        f"<p>Candidate: {html.escape(summary.candidate_name)}<br>"
        f"Overall score: {summary.overall_score}</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>Task</th><th>Score</th><th>Success Rate</th><th>Errors</th></tr>"
        f"{rows}</table>"
        + (f"<h2>Warnings</h2><ul>{warnings}</ul>" if warnings else "")
        + "</body></html>"
    )
