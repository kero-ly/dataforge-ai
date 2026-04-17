#!/usr/bin/env python3
"""SynEval Step 6: Aggregate results and generate paper-ready figures/tables.

Reads evaluation results from syneval/results/evals/ and generates:
  - Table 1: Exp 1 single-dimension ablation (MT-Bench × AlpacaEval × IFEval)
  - Table 2: Exp 2 pairwise interaction effect heatmap data
  - Table 3: Exp 6 baseline comparison
  - Figure 1: Bar chart of dimension contributions (delta from full)
  - Figure 2: Heatmap of similarity boundary (tau_min × tau_max)
  - Figure 3: Pareto curve (data retention rate vs MT-Bench)
  - Figure 4: Interaction effect heatmap matrix

Usage::

    python syneval/scripts/06_analyze_results.py \
        --results-dir syneval/results/evals \
        --output-dir syneval/figures
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available; skipping figures")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_results(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all eval result JSON files, grouped by benchmark type."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            benchmark = data.get("benchmark", path.stem.split("_")[0])
            grouped[benchmark].append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path, exc)
    return dict(grouped)


def aggregate_runs(
    results: list[dict[str, Any]],
    key: str = "overall_score",
) -> dict[str, dict[str, float]]:
    """Average scores across repeated runs (run_id 1, 2, 3) per config."""
    config_runs: dict[str, list[float]] = defaultdict(list)
    for r in results:
        config = r.get("config_name", "unknown")
        score = r.get(key)
        if score is not None:
            config_runs[config].append(float(score))
    return {
        config: {
            "mean": sum(scores) / len(scores),
            "std": (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5
                   if len(scores) > 1 else 0.0,
            "n": len(scores),
        }
        for config, scores in config_runs.items()
    }


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------
def make_exp1_table(
    mt_bench: dict[str, dict[str, float]],
    ifeval: dict[str, dict[str, float]],
    winrate: dict[str, dict[str, float]],
    output_dir: Path,
) -> str:
    """Generate LaTeX table for Exp 1 (single-dimension ablation)."""
    exp1_configs = [
        ("exp1_full", "All Dims (Full)"),
        ("exp1_no_completeness", "w/o D1 Completeness"),
        ("exp1_no_length", "w/o D2 Length"),
        ("exp1_no_llm_score", "w/o D3 LLM Score"),
        ("exp1_no_similarity", "w/o D4 Similarity"),
        ("exp1_no_dedup", "w/o D5 Dedup"),
        ("exp1_all_off", "No Filtering"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Single-Dimension Ablation Study (Exp 1). Scores are mean $\pm$ std across 3 runs.}",
        r"\label{tab:exp1-ablation}",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Configuration & MT-Bench & Win Rate & IFEval Acc \\",
        r" & (1--10) & (\%) & (\%) \\",
        r"\midrule",
    ]

    full_mt = mt_bench.get("exp1_full", {}).get("mean", 0.0)
    for cfg_key, cfg_label in exp1_configs:
        mt = mt_bench.get(cfg_key, {})
        ifeq = ifeval.get(cfg_key, {})
        wr = winrate.get(cfg_key, {})
        mt_str = f"{mt.get('mean', 0.0):.2f} $\\pm$ {mt.get('std', 0.0):.2f}" if mt else "---"
        wr_str = f"{wr.get('mean', 0.0)*100:.1f}" if wr else "---"
        if_str = f"{ifeq.get('mean', 0.0)*100:.1f}" if ifeq else "---"
        # Bold the best row (full pipeline)
        if cfg_key == "exp1_full":
            lines.append(f"  \\textbf{{{cfg_label}}} & \\textbf{{{mt_str}}} & \\textbf{{{wr_str}}} & \\textbf{{{if_str}}} \\\\")
        else:
            delta = mt.get("mean", full_mt) - full_mt if mt else 0.0
            delta_str = f" ({delta:+.2f})" if mt else ""
            lines.append(f"  {cfg_label} & {mt_str}{delta_str} & {wr_str} & {if_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)

    out_path = output_dir / "table_exp1_ablation.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    logger.info("Saved Exp 1 LaTeX table to %s", out_path)
    return table


def make_exp6_baseline_table(
    mt_bench: dict[str, dict[str, float]],
    ifeval: dict[str, dict[str, float]],
    output_dir: Path,
) -> str:
    """Generate LaTeX table for Exp 6 (baseline comparison)."""
    baselines = [
        ("exp6_best", "Ours (Best Config)"),
        ("exp6_alpagasus", "AlpaGasus (LLM Score Only)"),
        ("exp6_random_50pct", "Random 50% Subset"),
        ("exp6_length_only", "Length Filter Only"),
        ("exp6_ifd_style", "IFD-style (No LLM)"),
        ("exp1_all_off", "No Filtering"),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison with baselines (Exp 6).}",
        r"\label{tab:exp6-baselines}",
        r"\begin{tabular}{l c c}",
        r"\toprule",
        r"Method & MT-Bench & IFEval Acc \\",
        r"\midrule",
    ]

    for cfg_key, label in baselines:
        mt = mt_bench.get(cfg_key, {})
        ifeq = ifeval.get(cfg_key, {})
        mt_str = f"{mt.get('mean', 0.0):.2f}" if mt else "---"
        if_str = f"{ifeq.get('mean', 0.0)*100:.1f}" if ifeq else "---"
        if cfg_key == "exp6_best":
            lines.append(f"  \\textbf{{{label}}} & \\textbf{{{mt_str}}} & \\textbf{{{if_str}}} \\\\")
        else:
            lines.append(f"  {label} & {mt_str} & {if_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)
    out_path = output_dir / "table_exp6_baselines.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    logger.info("Saved Exp 6 table to %s", out_path)
    return table


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def plot_dimension_contributions(
    mt_bench: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Bar chart: how much each dimension contributes to MT-Bench score."""
    if not HAS_MATPLOTLIB:
        return

    dims = ["completeness", "length", "llm_score", "similarity", "dedup"]
    dim_labels = ["D1 Completeness", "D2 Length", "D3 LLM Score", "D4 Similarity", "D5 Dedup"]
    full_score = mt_bench.get("exp1_full", {}).get("mean", 0.0)

    contributions = []
    for dim in dims:
        no_dim = mt_bench.get(f"exp1_no_{dim}", {}).get("mean", full_score)
        contributions.append(full_score - no_dim)  # positive = this dim helps

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d32f2f" if c < 0 else "#1976d2" for c in contributions]
    bars = ax.bar(dim_labels, contributions, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MT-Bench Score Delta (contribution)")
    ax.set_title("Quality Dimension Contributions to SFT Performance")
    ax.set_xticklabels(dim_labels, rotation=20, ha="right", fontsize=9)

    for bar, val in zip(bars, contributions):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (0.02 if val >= 0 else -0.05),
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "fig_dimension_contributions.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dimension contributions chart to %s", out_path)


def plot_similarity_heatmap(
    mt_bench_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """2D heatmap: (tau_min, tau_max) → MT-Bench score (Exp 3)."""
    if not HAS_MATPLOTLIB:
        return

    tau_min_vals = [0.1, 0.2, 0.3, 0.4]
    tau_max_vals = [0.80, 0.85, 0.90, 0.95]

    # Build score matrix
    score_map: dict[tuple[float, float], float] = {}
    for r in mt_bench_results:
        name = r.get("config_name", "")
        if name.startswith("exp3_sim_") and "_" in name[9:]:
            parts = name[9:].split("_")
            if len(parts) == 2:
                try:
                    tmin = int(parts[0]) / 100.0
                    tmax = int(parts[1]) / 100.0
                    score_map[(tmin, tmax)] = r.get("overall_score", 0.0)
                except ValueError:
                    pass

    if not score_map:
        logger.info("No exp3 sim boundary results found; skipping heatmap")
        return

    matrix = np.zeros((len(tau_min_vals), len(tau_max_vals)))
    for i, tmin in enumerate(tau_min_vals):
        for j, tmax in enumerate(tau_max_vals):
            matrix[i, j] = score_map.get((tmin, tmax), float("nan"))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="MT-Bench Score")
    ax.set_xticks(range(len(tau_max_vals)))
    ax.set_yticks(range(len(tau_min_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in tau_max_vals])
    ax.set_yticklabels([f"{v:.2f}" for v in tau_min_vals])
    ax.set_xlabel("τ_max (upper similarity bound)")
    ax.set_ylabel("τ_min (lower similarity bound)")
    ax.set_title("Semantic Similarity Dual-Boundary (MT-Bench)")

    for i in range(len(tau_min_vals)):
        for j in range(len(tau_max_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "fig_similarity_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved similarity heatmap to %s", out_path)


def plot_pareto_curve(
    mt_bench_results: list[dict[str, Any]],
    subset_catalogue_path: Path | None,
    output_dir: Path,
) -> None:
    """Pareto curve: data retention rate vs MT-Bench score (Exp 4)."""
    if not HAS_MATPLOTLIB:
        return

    # Load subset retention rates
    retention: dict[str, float] = {}
    if subset_catalogue_path and subset_catalogue_path.exists():
        with open(subset_catalogue_path, encoding="utf-8") as f:
            catalogue = json.load(f)
        for s in catalogue.get("subsets", []):
            retention[s["config_name"]] = s.get("retention_rate", 1.0)

    points: list[tuple[float, float]] = []
    thresholds = [20, 30, 40, 50, 60, 70, 80, 90]
    for t in thresholds:
        cfg = f"exp4_threshold_{t}"
        score_data = next((r for r in mt_bench_results if r.get("config_name") == cfg), None)
        if score_data:
            rate = retention.get(cfg, 1.0)
            score = score_data.get("overall_score", 0.0)
            points.append((rate * 100, score, t))

    if not points:
        logger.info("No exp4 threshold results found; skipping Pareto curve")
        return

    points.sort(key=lambda x: x[0])
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    labels = [str(p[2]) for p in points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, "o-", color="#1976d2", linewidth=2, markersize=6)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(f"t={label}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Data Retention Rate (%)")
    ax.set_ylabel("MT-Bench Score")
    ax.set_title("Quality Threshold Pareto Curve (Exp 4)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "fig_pareto_curve.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Pareto curve to %s", out_path)


def plot_interaction_heatmap(
    mt_bench: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Heatmap of pairwise dimension interaction effects (Exp 2)."""
    if not HAS_MATPLOTLIB:
        return

    dims = ["completeness", "length", "llm_score", "similarity", "dedup"]
    n = len(dims)
    matrix = np.zeros((n, n))

    full_score = mt_bench.get("exp1_full", {}).get("mean", 0.0)
    base_score = mt_bench.get("exp1_all_off", {}).get("mean", 0.0)

    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if i == j:
                # Single dimension contribution
                solo = mt_bench.get(f"exp2_only_{d1}", {}).get("mean", base_score)
                matrix[i, j] = solo - base_score
            elif i < j:
                # Pairwise: interaction = joint - (solo_d1 + solo_d2 - base)
                joint_key = f"exp2_{d1}_{d2}"
                joint = mt_bench.get(joint_key, {}).get("mean", base_score)
                solo_d1 = mt_bench.get(f"exp2_only_{d1}", {}).get("mean", base_score)
                solo_d2 = mt_bench.get(f"exp2_only_{d2}", {}).get("mean", base_score)
                interaction = joint - (solo_d1 + solo_d2 - base_score)
                matrix[i, j] = matrix[j, i] = interaction

    short_labels = ["D1", "D2", "D3", "D4", "D5"]
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(matrix).max(), 0.01)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Interaction Effect (MT-Bench)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels)
    ax.set_yticklabels(short_labels)
    ax.set_title("Quality Dimension Interaction Effects (Exp 2)")

    full_labels = ["Completeness", "Length", "LLM Score", "Similarity", "Dedup"]
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    out_path = output_dir / "fig_interaction_heatmap.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved interaction heatmap to %s", out_path)


# ---------------------------------------------------------------------------
# Summary JSON for all configs
# ---------------------------------------------------------------------------
def write_results_summary(
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Write a unified summary CSV and JSON of all config × benchmark scores."""
    rows: list[dict[str, Any]] = []
    mt_results = all_results.get("mt_bench", [])
    if_results = all_results.get("ifeval", [])
    wr_results = all_results.get("alpacaeval_winrate", [])

    all_configs: set[str] = set()
    for r in mt_results + if_results + wr_results:
        all_configs.add(r.get("config_name", "unknown"))

    for config in sorted(all_configs):
        row: dict[str, Any] = {"config_name": config}
        mt_runs = [r for r in mt_results if r.get("config_name") == config]
        if_runs = [r for r in if_results if r.get("config_name") == config]
        wr_runs = [r for r in wr_results if r.get("config_name") == config]

        if mt_runs:
            scores = [r["overall_score"] for r in mt_runs if "overall_score" in r]
            row["mt_bench_mean"] = round(sum(scores) / len(scores), 4) if scores else None
            row["mt_bench_std"] = round(
                (sum((s - row["mt_bench_mean"]) ** 2 for s in scores) / len(scores)) ** 0.5, 4
            ) if len(scores) > 1 else 0.0

        if if_runs:
            scores = [r["strict_accuracy"] for r in if_runs if "strict_accuracy" in r]
            row["ifeval_mean"] = round(sum(scores) / len(scores), 4) if scores else None

        if wr_runs:
            scores = [r["win_rate"] for r in wr_runs if "win_rate" in r]
            row["winrate_mean"] = round(sum(scores) / len(scores), 4) if scores else None

        rows.append(row)

    # Write JSON summary
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    logger.info("Results summary written to %s", summary_path)

    # Write CSV
    if rows:
        csv_path = output_dir / "results_summary.csv"
        fields = ["config_name", "mt_bench_mean", "mt_bench_std", "ifeval_mean", "winrate_mean"]
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(k, "")) for k in fields) + "\n")
        logger.info("CSV summary written to %s", csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="SynEval Step 6: Aggregate results and generate figures/tables."
    )
    parser.add_argument("--results-dir", type=str,
                        default="syneval/results/evals",
                        help="Directory containing evaluation result JSON files.")
    parser.add_argument("--output-dir", type=str,
                        default="syneval/figures",
                        help="Directory to write figures and LaTeX tables.")
    parser.add_argument("--subset-catalogue", type=str, default=None,
                        help="Path to subset_catalogue.json (for Pareto curve retention rates).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(results_dir)
    logger.info("Loaded results: %s", {k: len(v) for k, v in all_results.items()})

    mt_bench_results = all_results.get("mt_bench", [])
    if_results = all_results.get("ifeval", [])
    wr_results = all_results.get("alpacaeval_winrate", [])

    mt_agg = aggregate_runs(mt_bench_results, "overall_score")
    if_agg = aggregate_runs(if_results, "strict_accuracy")
    wr_agg = aggregate_runs(wr_results, "win_rate")

    # Tables
    make_exp1_table(mt_agg, if_agg, wr_agg, output_dir)
    make_exp6_baseline_table(mt_agg, if_agg, output_dir)

    # Figures
    plot_dimension_contributions(mt_agg, output_dir)
    plot_similarity_heatmap(mt_bench_results, output_dir)
    catalogue_path = Path(args.subset_catalogue) if args.subset_catalogue else None
    plot_pareto_curve(mt_bench_results, catalogue_path, output_dir)
    plot_interaction_heatmap(mt_agg, output_dir)

    # Results summary
    write_results_summary(all_results, output_dir)

    logger.info("Analysis complete. Output in %s", output_dir)


if __name__ == "__main__":
    main()
