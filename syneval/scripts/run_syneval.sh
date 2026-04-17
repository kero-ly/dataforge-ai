#!/usr/bin/env bash
# SynEval Full Pipeline Orchestration Script
# Paper: Not All Quality Dimensions Matter Equally
#
# Runs the complete pipeline on the remote 8×4090 server:
#   Step 1: Synthesize 50K records
#   Step 2: Score quality (5 dimensions)
#   Step 3: Generate filtered subsets (57 ablation configs)
#   Step 4: Fine-tune models on each subset (Qwen + LLaMA, 3 runs each)
#   Step 5: Evaluate on MT-Bench / IFEval / Win-rate benchmarks
#   Step 6: Aggregate results and generate figures/tables
#
# Usage:
#   # Full pipeline (takes ~2 weeks on 8×4090):
#   bash syneval/scripts/run_syneval.sh all
#
#   # Single step:
#   bash syneval/scripts/run_syneval.sh synthesize
#   bash syneval/scripts/run_syneval.sh score
#   bash syneval/scripts/run_syneval.sh subsets
#   bash syneval/scripts/run_syneval.sh finetune exp1_full qwen 1
#   bash syneval/scripts/run_syneval.sh finetune_all_exp1
#   bash syneval/scripts/run_syneval.sh analyze

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/syneval/scripts"
DATA_DIR="${REPO_ROOT}/syneval/data"
RESULTS_DIR="${REPO_ROOT}/syneval/results"
MODELS_DIR="${RESULTS_DIR}/models"
EVALS_DIR="${RESULTS_DIR}/evals"
FIGURES_DIR="${REPO_ROOT}/syneval/figures"
SEEDS_DIR="${REPO_ROOT}/experiments/seeds"

# Server config (adjust for your environment)
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORTS="${VLLM_PORTS:-8100 8101 8102 8103 8104 8105 8106 8107}"
VLLM_URL="${VLLM_URL:-http://${VLLM_HOST}:8100/v1}"
MODEL_SYNTH="${MODEL_SYNTH:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_QWEN="${MODEL_QWEN:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_LLAMA="${MODEL_LLAMA:-meta-llama/Meta-Llama-3-8B-Instruct}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"

# Python env
PYTHON="${PYTHON:-python3}"
VENV="${REPO_ROOT}/.venv/bin/activate"
if [ -f "${VENV}" ]; then
    source "${VENV}"
fi

# Number of LoRA training runs per config (for std dev)
NUM_RUNS=3

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

mkdir -p "${DATA_DIR}" "${DATA_DIR}/subsets" "${MODELS_DIR}" "${EVALS_DIR}" "${FIGURES_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Data synthesis
# ---------------------------------------------------------------------------
cmd_synthesize() {
    log "=== Step 1: Synthesizing 50K records ==="
    local seed_dataset="${SEEDS_DIR}/seeds_10k.jsonl"
    [ -f "${seed_dataset}" ] || die "Seed dataset not found: ${seed_dataset}"

    ${PYTHON} "${SCRIPTS_DIR}/01_synthesize.py" \
        --seed-dataset "${seed_dataset}" \
        --target 50000 \
        --vllm-url "${VLLM_URL}" \
        --model "${MODEL_SYNTH}" \
        --output-dir "${DATA_DIR}" \
        --concurrency 50

    log "Synthesis complete. Output: ${DATA_DIR}/synthesized_50k.jsonl"
}

# ---------------------------------------------------------------------------
# Step 2: Quality scoring
# ---------------------------------------------------------------------------
cmd_score() {
    log "=== Step 2: Scoring quality on 5 dimensions ==="
    local dataset="${DATA_DIR}/synthesized_50k.jsonl"
    [ -f "${dataset}" ] || die "Synthesized dataset not found: ${dataset}"

    local skip_sim=""
    [ -z "${OPENAI_API_KEY}" ] && skip_sim="--skip-similarity" && \
        log "WARNING: OPENAI_API_KEY not set; skipping D4 similarity scoring"

    ${PYTHON} "${SCRIPTS_DIR}/02_score_quality.py" \
        --dataset "${dataset}" \
        --output-dir "${DATA_DIR}" \
        --vllm-url "${VLLM_URL}" \
        --model "${MODEL_SYNTH}" \
        --openai-api-key "${OPENAI_API_KEY:-}" \
        ${skip_sim}

    log "Quality scoring complete. Output: ${DATA_DIR}/scored_50k.jsonl"
}

# ---------------------------------------------------------------------------
# Step 3: Generate filtered subsets
# ---------------------------------------------------------------------------
cmd_subsets() {
    log "=== Step 3: Generating filtered subsets ==="
    local scored="${DATA_DIR}/scored_50k.jsonl"
    [ -f "${scored}" ] || die "Scored dataset not found: ${scored}"

    ${PYTHON} "${SCRIPTS_DIR}/03_generate_subsets.py" \
        --scored-dataset "${scored}" \
        --output-dir "${DATA_DIR}/subsets" \
        --exp all

    log "Subsets generated. Catalogue: ${DATA_DIR}/subsets/subset_catalogue.json"
}

# ---------------------------------------------------------------------------
# Step 4: Fine-tune (single config)
# cmd_finetune <config_name> <qwen|llama> <run_id>
# ---------------------------------------------------------------------------
cmd_finetune() {
    local config_name="${1:-exp1_full}"
    local model_type="${2:-qwen}"
    local run_id="${3:-1}"

    local subset="${DATA_DIR}/subsets/${config_name}.jsonl"
    [ -f "${subset}" ] || die "Subset not found: ${subset}"

    local base_model
    case "${model_type}" in
        qwen)  base_model="${MODEL_QWEN}" ;;
        llama) base_model="${MODEL_LLAMA}" ;;
        *)     die "Unknown model type: ${model_type}" ;;
    esac

    local output_dir="${MODELS_DIR}/${config_name}_${model_type}_run${run_id}"
    log "Fine-tuning: config=${config_name} model=${model_type} run=${run_id}"

    # Use torchrun for multi-GPU training if 8 GPUs are available
    local ngpu
    ngpu=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

    if [ "${ngpu}" -ge 2 ]; then
        torchrun --nproc_per_node="${ngpu}" \
            "${SCRIPTS_DIR}/04_finetune.py" \
            --dataset "${subset}" \
            --base-model "${base_model}" \
            --output-dir "${output_dir}" \
            --config-name "${config_name}" \
            --run-id "${run_id}"
    else
        ${PYTHON} "${SCRIPTS_DIR}/04_finetune.py" \
            --dataset "${subset}" \
            --base-model "${base_model}" \
            --output-dir "${output_dir}" \
            --config-name "${config_name}" \
            --run-id "${run_id}"
    fi

    log "Fine-tuning complete: ${output_dir}/final"
}

# ---------------------------------------------------------------------------
# Step 4b: Fine-tune all Exp 1 configs (sequential to manage GPU memory)
# ---------------------------------------------------------------------------
cmd_finetune_all_exp1() {
    local model_type="${1:-qwen}"
    log "=== Step 4: Fine-tuning all Exp 1 configs (model=${model_type}) ==="

    local configs=(
        "exp1_full"
        "exp1_all_off"
        "exp1_no_completeness"
        "exp1_no_length"
        "exp1_no_llm_score"
        "exp1_no_similarity"
        "exp1_no_dedup"
    )

    for config in "${configs[@]}"; do
        for run_id in $(seq 1 ${NUM_RUNS}); do
            log "  Training: ${config} run=${run_id}"
            cmd_finetune "${config}" "${model_type}" "${run_id}"
        done
    done
    log "All Exp 1 fine-tuning complete"
}

# ---------------------------------------------------------------------------
# Step 5: Evaluate a single model
# cmd_evaluate <config_name> <qwen|llama> <run_id> <benchmark>
# ---------------------------------------------------------------------------
cmd_evaluate() {
    local config_name="${1:-exp1_full}"
    local model_type="${2:-qwen}"
    local run_id="${3:-1}"
    local benchmark="${4:-mt_bench}"

    local model_path="${MODELS_DIR}/${config_name}_${model_type}_run${run_id}/final"
    [ -d "${model_path}" ] || die "Model not found: ${model_path}"

    log "Evaluating: config=${config_name} model=${model_type} run=${run_id} benchmark=${benchmark}"

    local judge_flag=""
    [ -n "${OPENAI_API_KEY}" ] && judge_flag="--judge-api-key ${OPENAI_API_KEY}"

    ${PYTHON} "${SCRIPTS_DIR}/05_evaluate.py" \
        --model-path "${model_path}" \
        --benchmark "${benchmark}" \
        --config-name "${config_name}" \
        --run-id "${run_id}" \
        --judge-model "${JUDGE_MODEL}" \
        ${judge_flag} \
        --output-dir "${EVALS_DIR}"
}

# ---------------------------------------------------------------------------
# Step 5b: Evaluate all Exp 1 configs
# ---------------------------------------------------------------------------
cmd_evaluate_all_exp1() {
    local model_type="${1:-qwen}"
    log "=== Step 5: Evaluating all Exp 1 configs (model=${model_type}) ==="

    local configs=(
        "exp1_full" "exp1_all_off"
        "exp1_no_completeness" "exp1_no_length" "exp1_no_llm_score"
        "exp1_no_similarity" "exp1_no_dedup"
    )
    local benchmarks=("mt_bench" "ifeval")
    [ -n "${OPENAI_API_KEY}" ] && benchmarks+=("alpacaeval")

    for config in "${configs[@]}"; do
        for run_id in $(seq 1 ${NUM_RUNS}); do
            for bm in "${benchmarks[@]}"; do
                local model_dir="${MODELS_DIR}/${config}_${model_type}_run${run_id}/final"
                if [ -d "${model_dir}" ]; then
                    cmd_evaluate "${config}" "${model_type}" "${run_id}" "${bm}" || \
                        log "WARNING: evaluation failed for ${config} ${bm} run${run_id}"
                else
                    log "Skipping ${config} run${run_id}: model not found"
                fi
            done
        done
    done
    log "All Exp 1 evaluations complete"
}

# ---------------------------------------------------------------------------
# Step 6: Analyze results
# ---------------------------------------------------------------------------
cmd_analyze() {
    log "=== Step 6: Analyzing results and generating figures ==="

    local catalogue="${DATA_DIR}/subsets/subset_catalogue.json"
    local cat_flag=""
    [ -f "${catalogue}" ] && cat_flag="--subset-catalogue ${catalogue}"

    ${PYTHON} "${SCRIPTS_DIR}/06_analyze_results.py" \
        --results-dir "${EVALS_DIR}" \
        --output-dir "${FIGURES_DIR}" \
        ${cat_flag}

    log "Analysis complete. Figures and tables in: ${FIGURES_DIR}"
}

# ---------------------------------------------------------------------------
# Step 7a: Sample 300 records for human annotation
# ---------------------------------------------------------------------------
cmd_annotation_sample() {
    log "=== Step 7a: Sampling 300 records for human annotation ==="
    local scored="${DATA_DIR}/scored_50k.jsonl"
    [ -f "${scored}" ] || die "Scored dataset not found: ${scored}"

    ${PYTHON} "${SCRIPTS_DIR}/07_human_annotation.py" sample \
        --scored-dataset "${scored}" \
        --output-dir "${DATA_DIR}/annotation" \
        --n-samples 300 \
        --n-annotators 3

    log "Annotation CSVs written to ${DATA_DIR}/annotation/"
    log "Share annotator_1.csv, annotator_2.csv, annotator_3.csv with your annotators."
    log "Do NOT share annotation_samples.jsonl (contains LLM scores)."
}

# ---------------------------------------------------------------------------
# Step 7b: Compute IAA and LLM-Judge correlation (after annotation is done)
# ---------------------------------------------------------------------------
cmd_annotation_analyze() {
    log "=== Step 7b: Computing IAA and LLM-Judge correlation ==="
    local annotation_dir="${DATA_DIR}/annotation"
    [ -d "${annotation_dir}" ] || die "Annotation directory not found: ${annotation_dir}"

    ${PYTHON} "${SCRIPTS_DIR}/07_human_annotation.py" analyze \
        --annotation-dir "${annotation_dir}" \
        --output-dir "${FIGURES_DIR}"

    log "Annotation analysis written to ${FIGURES_DIR}/annotation_analysis.json"
    log "LaTeX reliability table: ${FIGURES_DIR}/table_annotation_reliability.tex"
}

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
cmd_all() {
    log "=== SynEval Full Pipeline ==="
    cmd_synthesize
    cmd_score
    cmd_subsets
    cmd_annotation_sample
    cmd_finetune_all_exp1 qwen
    cmd_evaluate_all_exp1 qwen
    cmd_analyze
    log "=== Full pipeline complete (run 'annotation_analyze' after human annotation) ==="
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
    all)                  cmd_all ;;
    synthesize)           cmd_synthesize ;;
    score)                cmd_score ;;
    subsets)              cmd_subsets ;;
    annotation_sample)    cmd_annotation_sample ;;
    annotation_analyze)   cmd_annotation_analyze ;;
    finetune)             cmd_finetune "$@" ;;
    finetune_all_exp1)    cmd_finetune_all_exp1 "$@" ;;
    evaluate)             cmd_evaluate "$@" ;;
    evaluate_all_exp1)    cmd_evaluate_all_exp1 "$@" ;;
    analyze)              cmd_analyze ;;
    *)
        echo "Usage: $0 {all|synthesize|score|subsets|annotation_sample|annotation_analyze|finetune|finetune_all_exp1|evaluate|evaluate_all_exp1|analyze}"
        echo ""
        echo "  all                              - Run full pipeline"
        echo "  synthesize                       - Step 1: Synthesize 50K records"
        echo "  score                            - Step 2: Score quality (5 dims)"
        echo "  subsets                          - Step 3: Generate 57 filtered subsets"
        echo "  annotation_sample                - Step 7a: Sample 300 records for human annotation"
        echo "  annotation_analyze               - Step 7b: Compute IAA + LLM-Judge correlation"
        echo "  finetune <cfg> <qwen|llama> <id> - Step 4: Fine-tune one config"
        echo "  finetune_all_exp1 [qwen|llama]   - Step 4: Fine-tune all Exp 1 configs"
        echo "  evaluate <cfg> <model> <id> <bm> - Step 5: Evaluate one model"
        echo "  evaluate_all_exp1 [qwen|llama]   - Step 5: Evaluate all Exp 1 configs"
        echo "  analyze                          - Step 6: Aggregate results & figures"
        echo ""
        echo "Environment variables:"
        echo "  VLLM_URL         vLLM server URL (default: http://localhost:8100/v1)"
        echo "  MODEL_SYNTH      Synthesis model (default: Qwen/Qwen2.5-7B-Instruct)"
        echo "  MODEL_QWEN       Qwen SFT base model"
        echo "  MODEL_LLAMA      LLaMA SFT base model"
        echo "  OPENAI_API_KEY   OpenAI API key (for similarity scoring & MT-Bench judge)"
        echo "  JUDGE_MODEL      GPT judge model (default: gpt-4o)"
        ;;
esac
