#!/usr/bin/env python3
"""SynEval Step 7: Human annotation support for LLM-Judge validation.

Workflow:
  1. Sample 300 records from the scored dataset (stratified by LLM score quartile).
  2. Export a CSV/JSONL annotation file for each of 3 annotators.
  3. After annotation, compute:
     - Inter-Annotator Agreement (IAA) via Krippendorff's alpha and Cohen's kappa
     - Spearman / Kendall correlation between LLM-Judge score and human scores
     - Per-sub-criterion correlation breakdown

Usage::

    # Step A: Sample & export annotation batches
    python syneval/scripts/07_human_annotation.py sample \
        --scored-dataset syneval/data/scored_50k.jsonl \
        --output-dir syneval/data/annotation \
        --n-samples 300 --n-annotators 3

    # Step B (after annotators fill CSVs): Compute IAA and LLM-Judge correlation
    python syneval/scripts/07_human_annotation.py analyze \
        --annotation-dir syneval/data/annotation \
        --output-dir syneval/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# Annotation criteria (mirrors D3 LLM-Judge sub-criteria)
CRITERIA = [
    "instruction_clarity",   # Is the instruction clear and well-formed?
    "response_quality",      # Is the response helpful and accurate?
    "response_completeness", # Does the response fully address the instruction?
    "overall_quality",       # Overall quality on 1-5 scale
]

SCORE_RANGE = (1, 5)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _get_llm_score(record: dict[str, Any]) -> float | None:
    meta = record.get("metadata", {})
    qs = meta.get("quality_scores", {})
    llm_result = qs.get("llm_score", {})
    raw = llm_result.get("score")
    return float(raw) if raw is not None else None


def stratified_sample(
    records: list[dict[str, Any]],
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample n records, stratified by LLM score quartile.

    Records without an LLM score fall into a 5th stratum.
    """
    rng = random.Random(seed)

    # Partition into quartiles based on LLM score
    scored = [(i, _get_llm_score(r)) for i, r in enumerate(records)]
    has_score = [(i, s) for i, s in scored if s is not None]
    no_score = [(i, None) for i, s in scored if s is None]

    has_score.sort(key=lambda x: x[1])  # type: ignore[arg-type]
    quartile_size = max(1, len(has_score) // 4)

    strata: list[list[int]] = []
    for q in range(4):
        start = q * quartile_size
        end = start + quartile_size if q < 3 else len(has_score)
        strata.append([i for i, _ in has_score[start:end]])
    strata.append([i for i, _ in no_score])

    # Sample proportionally from each stratum
    per_stratum = n // len(strata)
    remainder = n - per_stratum * len(strata)

    sampled_indices: list[int] = []
    for k, stratum in enumerate(strata):
        count = per_stratum + (1 if k < remainder else 0)
        count = min(count, len(stratum))
        sampled_indices.extend(rng.sample(stratum, count))

    # Shuffle final sample
    rng.shuffle(sampled_indices)
    return [records[i] for i in sampled_indices]


# ---------------------------------------------------------------------------
# Export annotation CSV
# ---------------------------------------------------------------------------

def _format_text_preview(record: dict[str, Any], max_chars: int = 500) -> tuple[str, str]:
    """Return (instruction, response) preview strings."""
    synth = record.get("synthetic_data") or record
    instruction = str(synth.get("instruction", "")).strip()
    response = str(synth.get("response", synth.get("output", ""))).strip()
    if len(instruction) > max_chars:
        instruction = instruction[:max_chars] + "..."
    if len(response) > max_chars:
        response = response[:max_chars] + "..."
    return instruction, response


def export_annotation_csv(
    samples: list[dict[str, Any]],
    output_path: Path,
    annotator_id: int,
) -> None:
    """Write a CSV file for one annotator to fill in scores."""
    fields = (
        ["record_id", "annotator_id", "instruction", "response"]
        + CRITERIA
        + ["notes"]
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for i, rec in enumerate(samples):
            instruction, response = _format_text_preview(rec)
            rec_id = rec.get("id", f"sample_{i:04d}")
            row: dict[str, Any] = {
                "record_id": rec_id,
                "annotator_id": annotator_id,
                "instruction": instruction,
                "response": response,
                "notes": "",
            }
            # Leave score columns empty for annotator to fill
            for criterion in CRITERIA:
                row[criterion] = ""
            writer.writerow(row)

    logger.info("Annotation CSV written to %s (%d records)", output_path, len(samples))


def export_annotation_jsonl(
    samples: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write a JSONL file with record IDs and LLM judge scores (ground truth side)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(samples):
            rec_id = rec.get("id", f"sample_{i:04d}")
            llm_score = _get_llm_score(rec)
            meta = rec.get("metadata", {})
            qs = meta.get("quality_scores", {})
            llm_details = qs.get("llm_score", {}).get("details", {})
            instruction, response = _format_text_preview(rec, max_chars=2000)
            entry = {
                "record_id": rec_id,
                "instruction": instruction,
                "response": response,
                "llm_score": llm_score,
                "llm_details": llm_details,
                "sft_readiness_score": meta.get("sft_readiness_score"),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Ground-truth JSONL written to %s", output_path)


# ---------------------------------------------------------------------------
# IAA and correlation analysis
# ---------------------------------------------------------------------------

def load_annotations(annotation_dir: Path, n_annotators: int = 3) -> dict[str, Any]:
    """Load all annotator CSVs and merge into a unified structure.

    Returns:
        {record_id: {annotator_id: {criterion: score, ...}, ...}, ...}
    """
    merged: dict[str, dict[int, dict[str, float]]] = {}

    for ann_id in range(1, n_annotators + 1):
        csv_path = annotation_dir / f"annotator_{ann_id}.csv"
        if not csv_path.exists():
            logger.warning("Annotator CSV not found: %s", csv_path)
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec_id = row.get("record_id", "")
                if not rec_id:
                    continue
                scores: dict[str, float] = {}
                for criterion in CRITERIA:
                    val = row.get(criterion, "").strip()
                    if val:
                        try:
                            score = float(val)
                            if SCORE_RANGE[0] <= score <= SCORE_RANGE[1]:
                                scores[criterion] = score
                        except ValueError:
                            pass
                if scores:
                    merged.setdefault(rec_id, {})[ann_id] = scores

    return merged


def load_llm_scores(annotation_dir: Path) -> dict[str, float]:
    """Load LLM judge scores from the ground-truth JSONL."""
    llm_scores: dict[str, float] = {}
    gt_path = annotation_dir / "annotation_samples.jsonl"
    if not gt_path.exists():
        logger.warning("Ground-truth JSONL not found: %s", gt_path)
        return llm_scores

    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("llm_score") is not None:
                llm_scores[entry["record_id"]] = float(entry["llm_score"])

    return llm_scores


def _spearman_corr(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_vals[j]] == vals[sorted_vals[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_vals[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denom = (
        sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
        * sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
    )
    return num / denom if denom > 0 else float("nan")


def _kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Compute Kendall's tau-b."""
    n = len(xs)
    if n < 2:
        return float("nan")

    concordant = discordant = 0
    tied_x = tied_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0:
                    tied_x += 1
                if dy == 0:
                    tied_y += 1

    pairs = n * (n - 1) / 2
    denom = ((pairs - tied_x) * (pairs - tied_y)) ** 0.5
    return (concordant - discordant) / denom if denom > 0 else float("nan")


def _cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Compute Cohen's kappa for two raters (integer scores 1-5)."""
    n = len(rater1)
    if n == 0:
        return float("nan")

    categories = list(range(SCORE_RANGE[0], SCORE_RANGE[1] + 1))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(rater1, rater2):
        if a in cat_idx and b in cat_idx:
            conf[cat_idx[a]][cat_idx[b]] += 1

    total = sum(conf[i][j] for i in range(k) for j in range(k))
    if total == 0:
        return float("nan")

    p_o = sum(conf[i][i] for i in range(k)) / total
    row_sums = [sum(conf[i]) for i in range(k)]
    col_sums = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (total ** 2)

    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else float("nan")


def _krippendorff_alpha(data: list[list[float | None]]) -> float:
    """Compute Krippendorff's alpha for ordinal data.

    Args:
        data: list of [rater1_score, rater2_score, ...] per item.
              None means missing.
    """
    # Collect all paired disagreements
    n_raters = max(len(row) for row in data) if data else 0
    if n_raters < 2:
        return float("nan")

    # Observed disagreement
    o_num = 0.0
    o_den = 0.0
    for row in data:
        vals = [v for v in row if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                diff = vals[i] - vals[j]
                o_num += diff * diff
        o_den += m - 1

    # Expected disagreement
    all_vals = [v for row in data for v in row if v is not None]
    n = len(all_vals)
    if n < 2 or o_den == 0:
        return float("nan")

    e_num = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = all_vals[i] - all_vals[j]
            e_num += diff * diff
    e = e_num / (n - 1)

    if e == 0:
        return 1.0

    return 1.0 - (o_num / o_den) / e


def compute_iaa(
    annotations: dict[str, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    """Compute IAA metrics across all criteria and annotators."""
    results: dict[str, Any] = {}
    annotator_ids = sorted(
        {ann_id for rec in annotations.values() for ann_id in rec}
    )

    for criterion in CRITERIA:
        # Collect per-item scores for each annotator
        item_scores: list[list[float | None]] = []
        for rec_id, ann_scores in annotations.items():
            row: list[float | None] = []
            for ann_id in annotator_ids:
                score = ann_scores.get(ann_id, {}).get(criterion)
                row.append(score)
            item_scores.append(row)

        alpha = _krippendorff_alpha(item_scores)

        # Pairwise kappas
        pairwise_kappas: dict[str, float] = {}
        for i, a1 in enumerate(annotator_ids):
            for a2 in annotator_ids[i + 1:]:
                r1 = [
                    int(round(row[annotator_ids.index(a1)]))
                    for row in item_scores
                    if row[annotator_ids.index(a1)] is not None
                    and row[annotator_ids.index(a2)] is not None
                ]
                r2 = [
                    int(round(row[annotator_ids.index(a2)]))
                    for row in item_scores
                    if row[annotator_ids.index(a1)] is not None
                    and row[annotator_ids.index(a2)] is not None
                ]
                kappa = _cohens_kappa(r1, r2)
                pairwise_kappas[f"{a1}_vs_{a2}"] = round(kappa, 4)

        results[criterion] = {
            "krippendorff_alpha": round(alpha, 4) if alpha == alpha else None,
            "pairwise_cohen_kappa": pairwise_kappas,
            "n_items": len(item_scores),
        }

    return results


def compute_llm_human_correlation(
    annotations: dict[str, dict[int, dict[str, float]]],
    llm_scores: dict[str, float],
) -> dict[str, Any]:
    """Compute Spearman/Kendall correlation between LLM judge and mean human scores."""
    results: dict[str, Any] = {}

    for criterion in CRITERIA:
        pairs: list[tuple[float, float]] = []
        for rec_id, ann_scores in annotations.items():
            if rec_id not in llm_scores:
                continue
            human_vals = [
                scores[criterion]
                for scores in ann_scores.values()
                if criterion in scores
            ]
            if not human_vals:
                continue
            mean_human = sum(human_vals) / len(human_vals)
            llm_val = llm_scores[rec_id]
            pairs.append((llm_val, mean_human))

        if len(pairs) < 2:
            results[criterion] = {"n": len(pairs), "spearman": None, "kendall": None}
            continue

        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        spearman = _spearman_corr(xs, ys)
        kendall = _kendall_tau(xs, ys)
        results[criterion] = {
            "n": len(pairs),
            "spearman": round(spearman, 4) if spearman == spearman else None,
            "kendall": round(kendall, 4) if kendall == kendall else None,
        }

    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading scored dataset from %s", args.scored_dataset)
    records: list[dict[str, Any]] = []
    with open(args.scored_dataset, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records", len(records))

    samples = stratified_sample(records, args.n_samples, seed=args.seed)
    logger.info("Sampled %d records (stratified by LLM score)", len(samples))

    # Assign stable IDs to samples
    for i, rec in enumerate(samples):
        if "id" not in rec:
            rec["id"] = f"anno_{i:04d}"

    # Export ground-truth JSONL (for correlation analysis)
    export_annotation_jsonl(samples, output_dir / "annotation_samples.jsonl")

    # Export one CSV per annotator (blank score columns)
    for ann_id in range(1, args.n_annotators + 1):
        csv_path = output_dir / f"annotator_{ann_id}.csv"
        export_annotation_csv(samples, csv_path, annotator_id=ann_id)

    # Write sampling metadata
    meta = {
        "total_records": len(records),
        "n_samples": len(samples),
        "n_annotators": args.n_annotators,
        "seed": args.seed,
        "criteria": CRITERIA,
        "score_range": list(SCORE_RANGE),
        "instructions": (
            "Fill in scores 1-5 for each criterion. "
            "1=very poor, 3=acceptable, 5=excellent. "
            "Leave 'notes' for any comments."
        ),
    }
    with open(output_dir / "sampling_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "Annotation files written to %s\n"
        "  - annotation_samples.jsonl  (ground truth, do NOT share with annotators)\n"
        "  - annotator_1.csv ... annotator_%d.csv  (blank annotation sheets)",
        output_dir,
        args.n_annotators,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = annotation_dir / "sampling_meta.json"
    n_annotators = 3
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        n_annotators = meta.get("n_annotators", 3)

    logger.info("Loading annotation CSVs from %s", annotation_dir)
    annotations = load_annotations(annotation_dir, n_annotators=n_annotators)
    logger.info("Loaded annotations for %d records", len(annotations))

    if not annotations:
        logger.error("No annotation data found. Have the annotators filled in their CSVs?")
        sys.exit(1)

    llm_scores = load_llm_scores(annotation_dir)
    logger.info("Loaded LLM judge scores for %d records", len(llm_scores))

    # IAA
    iaa_results = compute_iaa(annotations)
    logger.info("IAA results:")
    for criterion, stats in iaa_results.items():
        alpha = stats["krippendorff_alpha"]
        logger.info(
            "  %-25s  Krippendorff alpha=%.3f  kappas=%s",
            criterion,
            alpha if alpha is not None else float("nan"),
            {k: f"{v:.3f}" for k, v in stats["pairwise_cohen_kappa"].items()},
        )

    # LLM-Human correlation
    corr_results = compute_llm_human_correlation(annotations, llm_scores)
    logger.info("LLM-Judge vs Human correlation:")
    for criterion, stats in corr_results.items():
        logger.info(
            "  %-25s  n=%d  Spearman=%.3f  Kendall=%.3f",
            criterion,
            stats["n"],
            stats["spearman"] if stats["spearman"] is not None else float("nan"),
            stats["kendall"] if stats["kendall"] is not None else float("nan"),
        )

    # Write analysis output
    analysis = {
        "iaa": iaa_results,
        "llm_human_correlation": corr_results,
        "n_annotated_records": len(annotations),
        "n_llm_scored_records": len(llm_scores),
    }
    out_path = output_dir / "annotation_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    logger.info("Annotation analysis written to %s", out_path)

    # Generate a simple LaTeX table for the paper
    _write_latex_correlation_table(corr_results, iaa_results, output_dir)


def _write_latex_correlation_table(
    corr: dict[str, Any],
    iaa: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write a LaTeX table summarizing LLM-Judge vs human correlation."""
    criterion_labels = {
        "instruction_clarity": "Instr. Clarity",
        "response_quality": "Resp. Quality",
        "response_completeness": "Completeness",
        "overall_quality": "Overall Quality",
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LLM-Judge reliability: Spearman/Kendall correlation with human annotations "
        r"(300 samples, 3 annotators). IAA = Krippendorff's $\alpha$.}",
        r"\label{tab:llm-judge-reliability}",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Criterion & Spearman $\rho$ & Kendall $\tau$ & IAA ($\alpha$) \\",
        r"\midrule",
    ]

    for criterion in CRITERIA:
        c_data = corr.get(criterion, {})
        i_data = iaa.get(criterion, {})
        label = criterion_labels.get(criterion, criterion)
        rho = c_data.get("spearman")
        tau = c_data.get("kendall")
        alpha = i_data.get("krippendorff_alpha")
        rho_str = f"{rho:.3f}" if rho is not None else "---"
        tau_str = f"{tau:.3f}" if tau is not None else "---"
        alpha_str = f"{alpha:.3f}" if alpha is not None else "---"
        lines.append(f"  {label} & {rho_str} & {tau_str} & {alpha_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    table = "\n".join(lines)

    out_path = output_dir / "table_annotation_reliability.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    logger.info("LaTeX reliability table written to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SynEval Step 7: Human annotation sampling and IAA analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- sample --
    sp_sample = subparsers.add_parser(
        "sample",
        help="Sample records and export annotation CSVs.",
    )
    sp_sample.add_argument(
        "--scored-dataset", type=str, required=True,
        help="Path to quality-scored JSONL (from 02_score_quality.py).",
    )
    sp_sample.add_argument(
        "--output-dir", type=str,
        default="syneval/data/annotation",
        help="Directory to write annotation CSVs.",
    )
    sp_sample.add_argument(
        "--n-samples", type=int, default=300,
        help="Number of records to sample (default: 300).",
    )
    sp_sample.add_argument(
        "--n-annotators", type=int, default=3,
        help="Number of annotators (default: 3).",
    )
    sp_sample.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling.",
    )

    # -- analyze --
    sp_analyze = subparsers.add_parser(
        "analyze",
        help="Compute IAA and LLM-Judge correlation from filled annotation CSVs.",
    )
    sp_analyze.add_argument(
        "--annotation-dir", type=str,
        default="syneval/data/annotation",
        help="Directory containing filled annotator CSVs.",
    )
    sp_analyze.add_argument(
        "--output-dir", type=str,
        default="syneval/figures",
        help="Directory to write analysis results and LaTeX table.",
    )

    args = parser.parse_args()

    if args.command == "sample":
        cmd_sample(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
