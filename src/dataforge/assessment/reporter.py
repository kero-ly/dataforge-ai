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

from dataforge.assessment.schema import DatasetAssessmentSummary, RecordAssessment


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
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def _timestamp_slug() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_assessment_report(
    *,
    run_name: str,
    suite_name: str,
    source_path: str,
    summary: DatasetAssessmentSummary,
    records: list[RecordAssessment],
    output_dir: str | Path,
    output_formats: list[str],
    persist_record_results: bool,
    config_snapshot: dict[str, Any],
) -> Path:
    root = Path(output_dir) / run_name / _timestamp_slug()
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": run_name,
        "suite": suite_name,
        "source_path": source_path,
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
    (root / "summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if persist_record_results:
        with open(root / "record_assessments.jsonl", "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")

    if "md" in output_formats:
        (root / "report.md").write_text(_render_markdown(summary), encoding="utf-8")
    if "html" in output_formats:
        (root / "report.html").write_text(_render_html(summary), encoding="utf-8")
    return root


def _render_markdown(summary: DatasetAssessmentSummary) -> str:
    lines = [
        f"# Assessment Report: {summary.suite}",
        "",
        f"- Total records: {summary.total_records}",
        f"- Sampled records: {summary.sampled_records}",
        f"- Sample seed: {summary.sample_seed}",
        f"- Overall quality score: {summary.overall_quality_score}",
        "",
        "## Evaluators",
        "",
        "| Evaluator | Pass Rate | Avg Score | Failed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for evaluator in summary.evaluator_summaries:
        avg_score = "--" if evaluator.avg_score is None else f"{evaluator.avg_score:.2f}"
        lines.append(
            f"| {evaluator.evaluator} | {evaluator.pass_rate:.2%} | {avg_score} | {evaluator.failed} |"
        )
    lines.extend(["", "## Dataset Metrics", ""])
    for key, value in sorted(summary.dataset_metrics.items()):
        lines.append(f"- {key}: {value}")
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in summary.warnings])
    return "\n".join(lines) + "\n"


def _render_html(summary: DatasetAssessmentSummary) -> str:
    rows = []
    for evaluator in summary.evaluator_summaries:
        avg_score = "--" if evaluator.avg_score is None else f"{evaluator.avg_score:.2f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(evaluator.evaluator)}</td>"
            f"<td>{evaluator.pass_rate:.2%}</td>"
            f"<td>{avg_score}</td>"
            f"<td>{evaluator.failed}</td>"
            "</tr>"
        )
    metric_items = "".join(
        f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
        for key, value in sorted(summary.dataset_metrics.items())
    )
    warning_items = "".join(
        f"<li>{html.escape(warning)}</li>" for warning in summary.warnings
    )
    return (
        "<html><body>"
        f"<h1>Assessment Report: {html.escape(summary.suite)}</h1>"
        f"<p>Total records: {summary.total_records}<br>"
        f"Sampled records: {summary.sampled_records}<br>"
        f"Overall quality score: {summary.overall_quality_score}</p>"
        "<h2>Evaluators</h2>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>Evaluator</th><th>Pass Rate</th><th>Avg Score</th><th>Failed</th></tr>"
        + "".join(rows)
        + "</table>"
        "<h2>Dataset Metrics</h2>"
        f"<ul>{metric_items}</ul>"
        + ("<h2>Warnings</h2><ul>" + warning_items + "</ul>" if warning_items else "")
        + "</body></html>"
    )
