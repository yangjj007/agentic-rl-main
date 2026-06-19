#!/usr/bin/env bash
# Smoke OPD from step 0: skip cold-start, SFT routing, and online-SFT slots.
# Uses opsd_only + disable_force_sft_replace (see config_opd_force_smoke.py).
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-scripts/test/config/config_opd_force_smoke.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"

# Short run defaults (override as needed)
export DYME_MAX_STEPS="${DYME_MAX_STEPS:-40}"
export DYME_MAX_TRAIN_SAMPLES="${DYME_MAX_TRAIN_SAMPLES:-64}"
export DYME_OPSD_MODE="${DYME_OPSD_MODE:-opsd_only}"
export DYME_OPSD_SKIP_DEGENERATE="${DYME_OPSD_SKIP_DEGENERATE:-0}"
export DYME_FAST_COLD_START_FRAC="${DYME_FAST_COLD_START_FRAC:-0}"

# Visual supervision off by default — faster and less VRAM for routing/OPD smoke
export DYME_VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-0}"
export DYME_VISUAL_REFINER="${DYME_VISUAL_REFINER:-0}"
export DYME_VISUAL_PREFETCH_IC="${DYME_VISUAL_PREFETCH_IC:-0}"

# Teacher on separate placement when possible
export DYME_TEACHER_DEVICE_MAP="${DYME_TEACHER_DEVICE_MAP:-auto}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

prepare_fast_test_data "${DYME_CONFIG}"
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"

echo "============================================================"
echo "OPD force smoke (no cold-start / no SFT routing)"
echo "config: ${DYME_CONFIG}"
echo "mode:   ${DYME_OPSD_MODE}"
echo "steps:  ${DYME_MAX_STEPS} | samples: ${DYME_MAX_TRAIN_SAMPLES}"
echo "visual: checker=${DYME_VISUAL_CHECKER} refiner=${DYME_VISUAL_REFINER}"
print_launch_plan

LOG_FILE="$(fast_train_log_path train_opd_force_smoke)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
