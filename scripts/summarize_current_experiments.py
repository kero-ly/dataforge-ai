#!/usr/bin/env python3
"""Build a paper-oriented summary from the currently available experiment results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_MD = RESULTS_DIR / "current_overall_experiment_comparison_20260310.md"
OUTPUT_JSON = RESULTS_DIR / "current_overall_experiment_comparison_20260310.json"


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.{digits}f}%"


def summarize_exp1() -> dict[str, Any]:
    files = sorted(RESULTS_DIR.glob("exp1_*.json"))
    valid: list[dict[str, Any]] = []
    deepseek_demo: list[dict[str, Any]] = []
    for path in files:
        row = load_json(path)
        row["_file"] = path.name
        if row.get("backend") == "deepseek":
            deepseek_demo.append(row)
            continue
        if row.get("backend") != "vllm":
            continue
        if row.get("dataset_size") not in {1000, 10000}:
            continue
        if row.get("completed") != row.get("dataset_size"):
            continue
        valid.append(row)

    best_by_key: dict[tuple[str | None, int | None], dict[str, Any]] = {}
    for row in valid:
        key = (row.get("method"), row.get("dataset_size"))
        prev = best_by_key.get(key)
        if prev is None or float(row.get("records_per_minute", 0.0)) > float(prev.get("records_per_minute", 0.0)):
            best_by_key[key] = row

    selected = sorted(best_by_key.values(), key=lambda row: (int(row.get("dataset_size", 0)), str(row.get("method"))))
    return {
        "rows": selected,
        "deepseek_demo_files": [row["_file"] for row in deepseek_demo],
    }


def summarize_exp3() -> dict[str, Any]:
    files = sorted((RESULTS_DIR / "exp3_100k_batch_final").glob("*.json"))
    deduped: dict[tuple[str | None, float | None], dict[str, Any]] = {}
    dropped: list[str] = []
    for path in files:
        row = load_json(path)
        row["_file"] = path.name
        key = (row.get("method"), row.get("kill_at_fraction"))
        prev = deduped.get(key)
        if prev is None or path.name > prev["_file"]:
            if prev is not None:
                dropped.append(prev["_file"])
            deduped[key] = row
        else:
            dropped.append(path.name)
    rows = sorted(deduped.values(), key=lambda row: (str(row.get("method")), float(row.get("kill_at_fraction", 0.0))))
    return {
        "rows": rows,
        "dropped_files": sorted(dropped),
    }


def summarize_exp4() -> dict[str, Any]:
    rows = load_json(RESULTS_DIR / "exp4_summary.json")
    rows = sorted(rows, key=lambda row: int(row.get("concurrency", 0)))
    best = max(rows, key=lambda row: float(row.get("throughput_records_per_min", 0.0)))
    return {
        "rows": rows,
        "best": best,
    }


def summarize_exp6() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS_DIR.glob("exp6_*.json")):
        row = load_json(path)
        if isinstance(row, dict) and row.get("config"):
            row["_file"] = path.name
            rows.append(row)
    baseline = next((row for row in rows if row.get("config") == "full"), None)
    deltas: list[dict[str, Any]] = []
    for row in rows:
        delta = dict(row)
        if baseline is not None and row is not baseline:
            delta["rpm_delta_vs_full"] = float(row.get("throughput_records_per_min", 0.0)) - float(
                baseline.get("throughput_records_per_min", 0.0)
            )
            delta["quality_delta_vs_full"] = float(row.get("avg_quality_score", 0.0)) - float(
                baseline.get("avg_quality_score", 0.0)
            )
        deltas.append(delta)
    return {
        "rows": deltas,
        "baseline": baseline,
    }


def summarize_exp9() -> dict[str, Any]:
    files = sorted((RESULTS_DIR / "exp9_scheduling_full").glob("*.json"))
    deduped: dict[tuple[str | None, int | None, int | None], dict[str, Any]] = {}
    dropped: list[str] = []
    for path in files:
        row = load_json(path)
        row["_file"] = path.name
        key = (row.get("schedule_name"), row.get("concurrency"), row.get("n_mutation_types"))
        prev = deduped.get(key)
        if prev is None or path.name > prev["_file"]:
            if prev is not None:
                dropped.append(prev["_file"])
            deduped[key] = row
        else:
            dropped.append(path.name)

    rows = list(deduped.values())
    by_schedule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_schedule[str(row.get("schedule_name"))].append(row)

    schedule_summary: list[dict[str, Any]] = []
    for schedule, members in sorted(by_schedule.items()):
        avg_rpm = sum(float(row.get("records_per_minute", 0.0)) for row in members) / len(members)
        avg_hit = sum(float(row.get("prefix_cache_hit_rate", 0.0)) for row in members) / len(members)
        best_row = max(members, key=lambda row: float(row.get("records_per_minute", 0.0)))
        schedule_summary.append(
            {
                "schedule_name": schedule,
                "num_configs": len(members),
                "avg_records_per_minute": avg_rpm,
                "avg_prefix_cache_hit_rate": avg_hit,
                "best_config": {
                    "concurrency": best_row.get("concurrency"),
                    "n_mutation_types": best_row.get("n_mutation_types"),
                    "records_per_minute": best_row.get("records_per_minute"),
                    "prefix_cache_hit_rate": best_row.get("prefix_cache_hit_rate"),
                    "file": best_row.get("_file"),
                },
            }
        )

    global_best = max(rows, key=lambda row: float(row.get("records_per_minute", 0.0)))
    return {
        "rows": sorted(rows, key=lambda row: (str(row.get("schedule_name")), int(row.get("concurrency", 0)), int(row.get("n_mutation_types", 0)))),
        "schedule_summary": schedule_summary,
        "global_best": global_best,
        "dropped_files": sorted(dropped),
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Current Overall Experiment Comparison (2026-03-10)")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- Available for comparison: `exp1`, `exp3`, `exp4`, `exp6`, `exp9`")
    lines.append("- Still missing / not paper-ready: large-scale API throughput, quality experiments (`exp10` / `Q1-Q3`), strengthened failure-injection ablation")
    lines.append("")

    lines.append("## Efficiency")
    lines.append("")
    lines.append("### Exp1 Throughput (best valid local runs)")
    lines.append("")
    lines.append("| Method | Backend | Dataset | RPM | File |")
    lines.append("|---|---:|---:|---:|---|")
    for row in summary["exp1"]["rows"]:
        lines.append(
            f"| {row.get('method')} | {row.get('backend')} | {row.get('dataset_size')} | "
            f"{fmt_float(float(row.get('records_per_minute', 0.0)))} | {row.get('_file')} |"
        )
    lines.append("")
    lines.append(f"- DeepSeek API is still not paper-usable: only demo files `{', '.join(summary['exp1']['deepseek_demo_files'])}` exist.")
    lines.append("")

    lines.append("### Exp9 Scheduling (deduped 100 configs)")
    lines.append("")
    lines.append("| Schedule | #Configs | Avg RPM | Avg Cache Hit | Best Config | Best RPM |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for row in summary["exp9"]["schedule_summary"]:
        best = row["best_config"]
        lines.append(
            f"| {row['schedule_name']} | {row['num_configs']} | {fmt_float(row['avg_records_per_minute'])} | "
            f"{fmt_pct(row['avg_prefix_cache_hit_rate'])} | c={best['concurrency']}, n={best['n_mutation_types']} | "
            f"{fmt_float(float(best['records_per_minute']))} |"
        )
    global_best = summary["exp9"]["global_best"]
    lines.append("")
    lines.append(
        f"- Global best scheduling config: `{global_best.get('schedule_name')}` / "
        f"`c={global_best.get('concurrency')}` / `n={global_best.get('n_mutation_types')}`, "
        f"`{fmt_float(float(global_best.get('records_per_minute', 0.0)))}` rpm, "
        f"`cache_hit={fmt_pct(float(global_best.get('prefix_cache_hit_rate', 0.0)))}`."
    )
    lines.append(
        f"- Dedup dropped files: `{', '.join(summary['exp9']['dropped_files'])}`."
    )
    lines.append("")

    lines.append("## Reliability")
    lines.append("")
    lines.append("### Exp3 100K Crash Recovery (deduped)")
    lines.append("")
    lines.append("| Method | Kill Point | Completed | Recovery Time (s) | RPM |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in summary["exp3"]["rows"]:
        lines.append(
            f"| {row.get('method')} | {fmt_pct(float(row.get('kill_at_fraction', 0.0)), 0)} | "
            f"{row.get('completed')} | {fmt_float(float(row.get('recovery_time_seconds', 0.0)))} | "
            f"{fmt_float(float(row.get('records_per_minute', 0.0)))} |"
        )
    lines.append("")
    lines.append(
        f"- Dropped duplicate files: `{', '.join(summary['exp3']['dropped_files'])}`."
    )
    lines.append("")

    lines.append("## Scalability")
    lines.append("")
    lines.append("### Exp4 Concurrency Scaling")
    lines.append("")
    lines.append("| Concurrency | RPM | p50 Latency (s) | p95 Latency (s) | p99 Latency (s) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in summary["exp4"]["rows"]:
        lines.append(
            f"| {row.get('concurrency')} | {fmt_float(float(row.get('throughput_records_per_min', 0.0)))} | "
            f"{fmt_float(float(row.get('p50_latency_s', 0.0)), 3)} | "
            f"{fmt_float(float(row.get('p95_latency_s', 0.0)), 3)} | "
            f"{fmt_float(float(row.get('p99_latency_s', 0.0)), 3)} |"
        )
    best = summary["exp4"]["best"]
    lines.append("")
    lines.append(
        f"- Best observed scalability point: `c={best.get('concurrency')}` with "
        f"`{fmt_float(float(best.get('throughput_records_per_min', 0.0)))}` rpm."
    )
    lines.append("")

    lines.append("## Ablation")
    lines.append("")
    lines.append("### Exp6 Mechanism Removal")
    lines.append("")
    lines.append("| Config | Completed | Rejected | RPM | Avg Quality | Delta RPM vs Full |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["exp6"]["rows"]:
        delta = row.get("rpm_delta_vs_full")
        lines.append(
            f"| {row.get('config')} | {row.get('completed')} | {row.get('rejected')} | "
            f"{fmt_float(float(row.get('throughput_records_per_min', 0.0)))} | "
            f"{fmt_float(float(row.get('avg_quality_score', 0.0)), 3)} | "
            f"{fmt_float(float(delta), 2) if delta is not None else '--'} |"
        )
    lines.append("")

    lines.append("## Current Takeaways")
    lines.append("")
    lines.append("- `exp3` now has paper-usable 100K recovery data for both `wal_jsonl` and `wal_sqlite` across 4 crash points.")
    lines.append("- `exp9` now has the full 100-config scheduling matrix; average schedule behavior can be compared immediately.")
    lines.append("- `exp1` is only partially paper-ready: vLLM local comparisons exist, but API-side throughput remains incomplete.")
    lines.append("- Quality experiments are still absent, so the paper currently has strong efficiency/reliability evidence but not the full quality axis.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = {
        "exp1": summarize_exp1(),
        "exp3": summarize_exp3(),
        "exp4": summarize_exp4(),
        "exp6": summarize_exp6(),
        "exp9": summarize_exp9(),
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUTPUT_MD.write_text(build_markdown(summary), encoding="utf-8")
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_MD}")


if __name__ == "__main__":
    main()
