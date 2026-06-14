#!/usr/bin/env bash
# Cross-model OPD (7B teacher + 0.5B student) with DeepSpeed ZeRO on ChartQA.
#
# Layout (2× GPU, recommended):
#   Each rank: student (ZeRO-sharded) + frozen 7B teacher on cuda:{LOCAL_RANK}
#
# Official refs:
#   https://huggingface.co/docs/transformers/deepspeed
#   https://huggingface.co/docs/accelerate/usage_guides/deepspeed
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-rlsd}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-}"
export DYME_OPSD_PRIVILEGE_PROFILE="${DYME_OPSD_PRIVILEGE_PROFILE:-text}"
export DYME_TEACHER_MODEL="${DYME_TEACHER_MODEL:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/opd-7b-chartqa-ds}"
# RLSD antidegen decode (also in config_rlsd; env overrides for A/B)
export DYME_MAX_COMPLETION_LENGTH="${DYME_MAX_COMPLETION_LENGTH:-128}"
export DYME_TEMPERATURE="${DYME_TEMPERATURE:-0.6}"
export DYME_REPETITION_PENALTY="${DYME_REPETITION_PENALTY:-1.35}"
export DYME_OPSD_DEGEN_WARMUP_STEPS="${DYME_OPSD_DEGEN_WARMUP_STEPS:-200}"
export DYME_SFT_WARMUP_STEPS="${DYME_SFT_WARMUP_STEPS:-200}"
export DYME_SFT_WARMUP_SLOTS="${DYME_SFT_WARMUP_SLOTS:-2}"
export DYME_FORMAT_MIN_THINKING="${DYME_FORMAT_MIN_THINKING:-8}"

# ZeRO-2 (default) or ZeRO-3 colocate for tighter memory:
#   ACCELERATE_CONFIG=default_config_zero3_colocate.yaml
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-default_config_zero2.yaml}"
# auto → colocate when DeepSpeed accelerate config is set (see opsd_utils/deepspeed_utils.py)
export DYME_TEACHER_DEVICE_MAP="${DYME_TEACHER_DEVICE_MAP:-auto}"
export DYME_OPSD_DETAIL_MIN_FREE_GB="${DYME_OPSD_DETAIL_MIN_FREE_GB:-4.0}"
export DYME_OPSD_DETAIL_EVERY="${DYME_OPSD_DETAIL_EVERY:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DYME_GRADIENT_CHECKPOINTING="${DYME_GRADIENT_CHECKPOINTING:-0}"
# ZeRO-1/2: one student forward/step (auto); ZeRO-3 or DDP may set DYME_GRADIENT_CHECKPOINTING=1.

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan
echo "DeepSpeed ZeRO OPD: ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
echo "Teacher placement: DYME_TEACHER_DEVICE_MAP=${DYME_TEACHER_DEVICE_MAP} (auto colocates under DeepSpeed)"

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_opd_7b_ds_$(date +%Y%m%d_%H%M%S).log"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config config/config_opd_7b_chartqa.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  2>&1 | tee "${LOG_FILE}"
