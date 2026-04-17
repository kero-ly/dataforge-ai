#!/usr/bin/env python3
"""SynEval Step 3: Generate 57 ablation filter configurations and apply them.

Configurations cover:
  Exp 1 - single-dimension ablation (7 configs)
  Exp 2 - pairwise interaction (10 pairs + all = 11 configs)
  Exp 3 - semantic similarity dual-boundary (16 configs)
  Exp 4 - quality threshold Pareto (8 configs)
  Exp 5 - cross-model (reuse Exp 1 best configs)
  Exp 6 - comparison baselines (5 configs)

Each config produces a filtered JSONL subset under syneval/data/subsets/.

Usage::

    python syneval/scripts/03_generate_subsets.py \
        --scored-dataset syneval/data/scored_50k.jsonl \
        --output-dir syneval/data/subsets \
        --exp all

    # Or generate only one experiment group:
    python syneval/scripts/03_generate_subsets.py \
        --scored-dataset syneval/data/scored_50k.jsonl \
        --output-dir syneval/data/subsets \
        --exp exp1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# Canonical 5 dimensions (keys into quality_scores)
DIMS = ["completeness", "length", "llm_score", "similarity", "dedup"]

# LLM score thresholds for filtering
LLM_SCORE_THRESHOLD = 3.5  # D3 pass threshold (1-5 scale)


# ---------------------------------------------------------------------------
# Config definition
# ---------------------------------------------------------------------------

def _dim_passes(quality_scores: dict[str, Any], dim: str, threshold: float | None = None) -> bool:
    """Return True if the record passes the given dimension filter."""
    result = quality_scores.get(dim, {})
    if threshold is not None and result.get("score") is not None:
        return float(result["score"]) >= threshold
    return bool(result.get("passed", False))


def _sim_passes(quality_scores: dict[str, Any], sim_min: float, sim_max: float) -> bool:
    result = quality_scores.get("similarity", {})
    score = result.get("score")
    if score is None:
        return False
    return sim_min <= float(score) <= sim_max


def _sft_readiness_passes(meta: dict[str, Any], threshold: float) -> bool:
    return float(meta.get("sft_readiness_score", 0.0)) >= threshold


def filter_by_config(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply a filter config dict to a list of scored records."""
    filtered = []
    for rec in records:
        meta = rec.get("metadata", {})
        qs = meta.get("quality_scores", {})

        passes = True

        # Dimension-level filters
        for dim in DIMS:
            if config.get(f"use_{dim}", False):
                if dim == "llm_score":
                    thresh = config.get("llm_threshold", LLM_SCORE_THRESHOLD)
                    if not _dim_passes(qs, "llm_score", threshold=thresh):
                        passes = False
                        break
                elif dim == "similarity":
                    sim_min = config.get("sim_min", 0.3)
                    sim_max = config.get("sim_max", 0.95)
                    if not _sim_passes(qs, sim_min, sim_max):
                        passes = False
                        break
                else:
                    if not _dim_passes(qs, dim):
                        passes = False
                        break

        # SFT readiness threshold filter (Exp 4)
        if passes and config.get("sft_threshold") is not None:
            if not _sft_readiness_passes(meta, config["sft_threshold"]):
                passes = False

        if passes:
            filtered.append(rec)

    return filtered


# ---------------------------------------------------------------------------
# Config catalogue
# ---------------------------------------------------------------------------

def get_exp1_configs() -> list[dict[str, Any]]:
    """Exp 1: single-dimension ablation. 7 configs."""
    full: dict[str, Any] = {"name": "exp1_full", "use_completeness": True, "use_length": True,
                             "use_llm_score": True, "use_similarity": True, "use_dedup": True,
                             "sim_min": 0.3, "sim_max": 0.95}
    all_off: dict[str, Any] = {"name": "exp1_all_off", "use_completeness": False, "use_length": False,
                                "use_llm_score": False, "use_similarity": False, "use_dedup": False}
    configs: list[dict[str, Any]] = [all_off, full]
    for dim in DIMS:
        cfg: dict[str, Any] = {
            "name": f"exp1_no_{dim}",
            "use_completeness": True, "use_length": True,
            "use_llm_score": True, "use_similarity": True, "use_dedup": True,
            "sim_min": 0.3, "sim_max": 0.95,
        }
        cfg[f"use_{dim}"] = False  # turn off this one dimension
        configs.append(cfg)
    return configs


def get_exp2_configs() -> list[dict[str, Any]]:
    """Exp 2: pairwise dimension combinations. 10 pairs + all-5 = 11 configs."""
    from itertools import combinations
    configs = []
    # all-5 already in exp1; add it here as well for standalone analysis
    for d1, d2 in combinations(DIMS, 2):
        cfg: dict[str, Any] = {
            "name": f"exp2_{d1}_{d2}",
            "use_completeness": False, "use_length": False,
            "use_llm_score": False, "use_similarity": False, "use_dedup": False,
            "sim_min": 0.3, "sim_max": 0.95,
        }
        cfg[f"use_{d1}"] = True
        cfg[f"use_{d2}"] = True
        configs.append(cfg)
    # Add each single dimension alone for interaction effect computation
    for dim in DIMS:
        cfg = {
            "name": f"exp2_only_{dim}",
            "use_completeness": False, "use_length": False,
            "use_llm_score": False, "use_similarity": False, "use_dedup": False,
            "sim_min": 0.3, "sim_max": 0.95,
        }
        cfg[f"use_{dim}"] = True
        configs.append(cfg)
    return configs


def get_exp3_configs() -> list[dict[str, Any]]:
    """Exp 3: similarity boundary sweep. 16 configs."""
    tau_min_values = [0.1, 0.2, 0.3, 0.4]
    tau_max_values = [0.80, 0.85, 0.90, 0.95]
    configs = []
    for tau_min in tau_min_values:
        for tau_max in tau_max_values:
            if tau_min >= tau_max:
                continue
            cfg: dict[str, Any] = {
                "name": f"exp3_sim_{int(tau_min*100)}_{int(tau_max*100)}",
                "use_completeness": True, "use_length": True,
                "use_llm_score": True, "use_similarity": True, "use_dedup": True,
                "sim_min": tau_min, "sim_max": tau_max,
            }
            configs.append(cfg)
    # Add single-boundary variants for comparison
    configs.append({
        "name": "exp3_sim_no_lower",
        "use_completeness": True, "use_length": True,
        "use_llm_score": True, "use_similarity": True, "use_dedup": True,
        "sim_min": 0.0, "sim_max": 0.95,
    })
    configs.append({
        "name": "exp3_sim_no_upper",
        "use_completeness": True, "use_length": True,
        "use_llm_score": True, "use_similarity": True, "use_dedup": True,
        "sim_min": 0.3, "sim_max": 1.0,
    })
    return configs


def get_exp4_configs() -> list[dict[str, Any]]:
    """Exp 4: SFT readiness threshold Pareto. 8 configs."""
    thresholds = [20, 30, 40, 50, 60, 70, 80, 90]
    configs = []
    for t in thresholds:
        configs.append({
            "name": f"exp4_threshold_{t}",
            "use_completeness": True, "use_length": True,
            "use_llm_score": True, "use_similarity": True, "use_dedup": True,
            "sim_min": 0.3, "sim_max": 0.95,
            "sft_threshold": float(t),
        })
    return configs


def get_exp6_configs() -> list[dict[str, Any]]:
    """Exp 6: baseline comparisons. 5 configs."""
    return [
        # AlpaGasus-style: LLM score only
        {"name": "exp6_alpagasus", "use_completeness": False, "use_length": False,
         "use_llm_score": True, "use_similarity": False, "use_dedup": False,
         "llm_threshold": 4.0},
        # Random 50% subset
        {"name": "exp6_random_50pct", "use_completeness": False, "use_length": False,
         "use_llm_score": False, "use_similarity": False, "use_dedup": False,
         "random_sample_rate": 0.5},
        # Length filter only
        {"name": "exp6_length_only", "use_completeness": False, "use_length": True,
         "use_llm_score": False, "use_similarity": False, "use_dedup": False},
        # IFD-style: completeness + length + dedup (no LLM)
        {"name": "exp6_ifd_style", "use_completeness": True, "use_length": True,
         "use_llm_score": False, "use_similarity": False, "use_dedup": True},
        # Best config from paper (filled in after Exp 1-4)
        {"name": "exp6_best", "use_completeness": True, "use_length": True,
         "use_llm_score": True, "use_similarity": True, "use_dedup": True,
         "sim_min": 0.3, "sim_max": 0.90, "sft_threshold": 50.0},
    ]


EXP_GROUPS = {
    "exp1": get_exp1_configs,
    "exp2": get_exp2_configs,
    "exp3": get_exp3_configs,
    "exp4": get_exp4_configs,
    "exp6": get_exp6_configs,
}


def get_all_configs() -> list[dict[str, Any]]:
    all_configs = []
    seen_names: set[str] = set()
    for fn in EXP_GROUPS.values():
        for cfg in fn():
            if cfg["name"] not in seen_names:
                all_configs.append(cfg)
                seen_names.add(cfg["name"])
    return all_configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def apply_and_save(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Filter records by config and write subset JSONL. Returns stats."""
    # Handle random sampling baseline
    sample_rate = config.get("random_sample_rate")
    if sample_rate is not None:
        k = int(len(records) * sample_rate)
        filtered = random.sample(records, k)
    else:
        filtered = filter_by_config(records, config)

    output_path = output_dir / f"{config['name']}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in filtered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "config_name": config["name"],
        "total_input": len(records),
        "total_output": len(filtered),
        "retention_rate": round(len(filtered) / max(len(records), 1), 4),
        "output_path": str(output_path),
    }
    logger.info(
        "Config %-40s  retained %5d / %5d  (%.1f%%)",
        config["name"],
        len(filtered),
        len(records),
        stats["retention_rate"] * 100,
    )
    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SynEval Step 3: Generate filtered data subsets for ablation."
    )
    parser.add_argument("--scored-dataset", type=str, required=True,
                        help="Path to quality-scored JSONL (from 02_score_quality.py).")
    parser.add_argument("--output-dir", type=str, default="syneval/data/subsets",
                        help="Directory to write filtered subsets.")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all"] + list(EXP_GROUPS.keys()),
                        help="Which experiment group configs to generate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random-sampling configs.")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scored records
    logger.info("Loading scored records from %s", args.scored_dataset)
    records: list[dict[str, Any]] = []
    with open(args.scored_dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records", len(records))

    # Get configs
    if args.exp == "all":
        configs = get_all_configs()
    else:
        configs = EXP_GROUPS[args.exp]()

    logger.info("Generating %d filter configurations...", len(configs))

    all_stats = []
    for config in configs:
        stats = apply_and_save(records, config, output_dir)
        all_stats.append(stats)

    # Save config catalogue and stats
    catalogue_path = output_dir / "subset_catalogue.json"
    with open(catalogue_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_configs": len(configs),
            "subsets": all_stats,
            "configs": configs,
        }, f, indent=2)
    logger.info("Subset catalogue written to %s", catalogue_path)


if __name__ == "__main__":
    main()
