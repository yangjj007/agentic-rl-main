#!/usr/bin/env bash
# 200-step smoke for OPD 7B + RLSD anti-collapse fixes (config inherit, format reward, warmup gates).
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_MAX_STEPS="${DYME_MAX_STEPS:-200}"
export DYME_OPSD_MODE="${DYME_OPSD_MODE:-rlsd}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/opd-7b-chartqa-smoke}"
export DYME_MAX_COMPLETION_LENGTH="${DYME_MAX_COMPLETION_LENGTH:-96}"
export DYME_TEMPERATURE="${DYME_TEMPERATURE:-0.5}"
export DYME_REPETITION_PENALTY="${DYME_REPETITION_PENALTY:-1.5}"
export DYME_OPSD_DEGEN_WARMUP_STEPS="${DYME_OPSD_DEGEN_WARMUP_STEPS:-200}"
export DYME_SFT_WARMUP_STEPS="${DYME_SFT_WARMUP_STEPS:-500}"
export DYME_SFT_WARMUP_SLOTS="${DYME_SFT_WARMUP_SLOTS:-4}"
export DYME_SFT_COLD_START_FRAC="${DYME_SFT_COLD_START_FRAC:-0.08}"
export DYME_FORMAT_MIN_THINKING="${DYME_FORMAT_MIN_THINKING:-8}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config_zero2.yaml}"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_opd_7b_smoke_$(date +%Y%m%d_%H%M%S).log"

echo "OPD 7B smoke: max_steps=${DYME_MAX_STEPS}, log=${LOG_FILE}"
echo "Success criteria (grep log after run):"
echo "  degenerate_rate < 0.5"
echo "  opsd_mask_true / batch > 0.08"
echo "  advantage_abs_mean > 0"
echo "  grad_norm > 0"
echo "  phase/sft_cold_start=1 during first ${DYME_SFT_COLD_START_FRAC} of steps"
echo "  logits/p_answer_first > 0.8 after cold start"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_opd_7b_chartqa.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
