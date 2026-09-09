#!/usr/bin/env bash
# =====================================================================
# Convenience wrapper around train_dft.py.
#
# Runs through `uv run` so the pinned environment from uv.lock is used.
# Every argument you pass is forwarded to train_dft.py, e.g.
#
#   ./train_dft.sh --model-path Qwen/Qwen2.5-3B-Instruct --dataset data/mine.jsonl
#   ./train_dft.sh --method orpo --num-train-epochs 3
#
# Environment:
#   MODEL_PATH             default base model when --model-path is not given
#   DATASET                default dataset when --dataset is not given
#   CUDA_VISIBLE_DEVICES   GPU to train on (default: 0)
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not installed. See https://docs.astral.sh/uv/getting-started/" >&2
    exit 1
fi

# Supply defaults only when the caller did not pass the flag itself.
# Matches both "--flag value" and "--flag=value", which argparse accepts alike.
ARGS=("$@")
has_flag() {
    local needle="$1" arg
    for arg in ${ARGS[@]+"${ARGS[@]}"}; do
        [[ "$arg" == "$needle" || "$arg" == "$needle="* ]] && return 0
    done
    return 1
}

if ! has_flag --model-path; then
    if [[ -z "${MODEL_PATH:-}" ]]; then
        echo "error: no model given. Pass --model-path <hf-id-or-dir> or set MODEL_PATH." >&2
        exit 1
    fi
    ARGS+=(--model-path "$MODEL_PATH")
fi
if ! has_flag --dataset && [[ -n "${DATASET:-}" ]]; then
    ARGS+=(--dataset "$DATASET")
fi

# Auto-detect the next free numbered log file so repeat runs do not clobber.
mkdir -p logs
LOG_NUM=1
while [[ -f "logs/train_dft_${LOG_NUM}.log" ]]; do
    LOG_NUM=$((LOG_NUM + 1))
done
LOG_FILE="logs/train_dft_${LOG_NUM}.log"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "logging to $LOG_FILE"
echo "running: uv run python train_dft.py ${ARGS[*]}"

uv run python train_dft.py ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
