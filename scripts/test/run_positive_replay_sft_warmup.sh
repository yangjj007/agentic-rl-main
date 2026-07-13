#!/usr/bin/env bash
# Offline SFT warmup on positive replay targets, then use final_checkpoint for DyME.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${TEST_DIR}/../.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/test/launch_utils.sh"

DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  bash scripts/test/run_positive_replay_sft_warmup.sh [--dry-run]

Environment:
  DYME_REPLAY_TRAIN_DATASET   replay_train.json path
  DYME_REPLAY_SFT_RUN_ID      output subdirectory name
  DYME_REPLAY_SFT_EPOCHS      SFT epochs, default 0.5
  DYME_REPLAY_SFT_OUTPUT_ROOT output root
  DYME_REPLAY_SFT_LOG_ROOT    log root
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export DYME_REPLAY_TRAIN_DATASET="${DYME_REPLAY_TRAIN_DATASET:-outputs/test-fast/positive-replay-buffer/student_hint_short_full/replay_train.json}"
export DYME_REPLAY_SFT_EPOCHS="${DYME_REPLAY_SFT_EPOCHS:-0.5}"
export DYME_REPLAY_SFT_OUTPUT_ROOT="${DYME_REPLAY_SFT_OUTPUT_ROOT:-outputs/test-fast/positive-replay-sft}"
export DYME_REPLAY_SFT_LOG_ROOT="${DYME_REPLAY_SFT_LOG_ROOT:-outputs/test-fast/logs/positive-replay-sft}"
export DYME_REPLAY_SFT_RUN_ID="${DYME_REPLAY_SFT_RUN_ID:-replay_sft_warmup_$(date +%Y%m%d_%H%M%S)}"
export DYME_SFT_OUTPUT_DIR="${DYME_SFT_OUTPUT_DIR:-${DYME_REPLAY_SFT_OUTPUT_ROOT}/${DYME_REPLAY_SFT_RUN_ID}}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"
STUDENT_MODEL="${DYME_STUDENT_MODEL:-/home/deepseek_VG/deepseek/models/llava-0.5b-ov}"

if [[ ! -f "${DYME_REPLAY_TRAIN_DATASET}" ]]; then
  echo "Replay train dataset not found: ${DYME_REPLAY_TRAIN_DATASET}" >&2
  echo "Generate it with scripts/analysis/export_positive_replay_buffer.py first." >&2
  exit 1
fi

NUM_PROCESSES="$(detect_num_gpus)"
CONFIG_PATH="scripts/test/config/config_positive_replay_sft.py"
FINAL_CKPT="${DYME_SFT_OUTPUT_DIR}/final_checkpoint"
LOG_FILE="${DYME_REPLAY_SFT_LOG_ROOT}/${DYME_REPLAY_SFT_RUN_ID}.log"

echo "============================================================"
echo "Positive replay SFT warmup"
echo "config: ${CONFIG_PATH}"
echo "DYME_REPLAY_TRAIN_DATASET=${DYME_REPLAY_TRAIN_DATASET}"
echo "DYME_REPLAY_SFT_EPOCHS=${DYME_REPLAY_SFT_EPOCHS}"
echo "DYME_SFT_OUTPUT_DIR=${DYME_SFT_OUTPUT_DIR}"
echo "student model: ${STUDENT_MODEL}"
echo "final checkpoint: ${FINAL_CKPT}"
echo "log file: ${LOG_FILE}"
print_launch_plan

echo "Command:"
echo "  accelerate launch --config_file ${ACCELERATE_CONFIG} --num_processes ${NUM_PROCESSES} main_sft.py --config ${CONFIG_PATH} --pretrained_model_path ${STUDENT_MODEL}"
echo "Next DyME env:"
echo "  DYME_STUDENT_MODEL=${FINAL_CKPT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry-run only."
  exit 0
fi

mkdir -p "${DYME_REPLAY_SFT_LOG_ROOT}" "${DYME_SFT_OUTPUT_DIR}"
run_train_with_log "${LOG_FILE}" \
  accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main_sft.py \
    --config "${CONFIG_PATH}" \
    --pretrained_model_path "${STUDENT_MODEL}"
