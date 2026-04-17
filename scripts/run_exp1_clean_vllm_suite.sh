#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${ROOT_DIR}/experiments/results/exp1_clean_vllm_${STAMP}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
BASE_URLS="${BASE_URLS:-http://localhost:8100/v1,http://localhost:8101/v1,http://localhost:8102/v1,http://localhost:8103/v1,http://localhost:8104/v1,http://localhost:8105/v1,http://localhost:8106/v1,http://localhost:8107/v1}"
CONCURRENCY="${CONCURRENCY:-100}"

mkdir -p "${OUT_DIR}"

run_case() {
  echo
  echo "[exp1-clean] $*"
  "${PYTHON_BIN}" -m experiments.exp1_throughput.run_throughput "$@"
}

COMMON_ARGS=(
  --backend vllm
  --concurrency "${CONCURRENCY}"
  --model "${MODEL}"
  --base-urls "${BASE_URLS}"
  --cluster-routing round_robin
  --scenario ideal
  --mode burst
  --max-retries 0
  --mutation-schedule random
  --output-dir "${OUT_DIR}"
)

# Primary apples-to-apples comparison for the paper:
# DataForge vs naive_asyncio on the same EvolInstruct workload.
run_case "${COMMON_ARGS[@]}" \
  --method dataforge \
  --dataset experiments/data/seeds_1k.jsonl

run_case "${COMMON_ARGS[@]}" \
  --method naive_asyncio \
  --workload-mode dataforge_strategy \
  --dataset experiments/data/seeds_1k.jsonl

run_case "${COMMON_ARGS[@]}" \
  --method dataforge \
  --dataset experiments/data/seeds_10k.jsonl

run_case "${COMMON_ARGS[@]}" \
  --method naive_asyncio \
  --workload-mode dataforge_strategy \
  --dataset experiments/data/seeds_10k.jsonl

# Reference floor, not a main baseline for the throughput claim.
run_case "${COMMON_ARGS[@]}" \
  --method sequential \
  --dataset experiments/data/seeds_1k.jsonl

echo
echo "[exp1-clean] results saved under ${OUT_DIR}"
