#!/usr/bin/env bash
# Cross-model OPD (7B teacher + 0.5B student) with DeepSpeed on ChartQA.
#
# Default: ZeRO-0 (no student sharding) when VRAM is sufficient — fastest on 8× H800.
#   Each rank: full student + frozen 7B teacher on cuda:{LOCAL_RANK}
#
# Memory-tight fallback:
#   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_chartqa_deepspeed.sh
#   ACCELERATE_CONFIG=default_config_zero3_colocate.yaml bash scripts/train_opd_7b_chartqa_deepspeed.sh
#
# Cold-start / decode / warmup defaults: config/config_rlsd_chartqa.py (inherited by
# config/config_opd_7b_chartqa.py). Override via DYME_* env only when needed.
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
# Local ModelScope/HF dirs: use absolute paths (``~`` is expanded in config).
#   export DYME_STUDENT_MODEL=/home/deepseek_VG/deepseek/models/llava-0.5b-ov
#   export DYME_TEACHER_MODEL=/home/deepseek_VG/deepseek/models/llava-7b-ov
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/opd-7b-chartqa-ds}"

# ZeRO-0 default (8 GPU → default_config_8gpu_deepspeed.yaml; else default_config_deepspeed.yaml)
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero0_config)}"
# auto → colocate when DeepSpeed accelerate config is set (see opsd_utils/deepspeed_utils.py)
export DYME_TEACHER_DEVICE_MAP="${DYME_TEACHER_DEVICE_MAP:-auto}"
export DYME_OPSD_DETAIL_MIN_FREE_GB="${DYME_OPSD_DETAIL_MIN_FREE_GB:-4.0}"
export DYME_OPSD_DETAIL_EVERY="${DYME_OPSD_DETAIL_EVERY:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DYME_GRADIENT_CHECKPOINTING="${DYME_GRADIENT_CHECKPOINTING:-0}"

# OPD text profile does not need real DePlot; placeholder vf_full is enough and fast.
export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-0}"
ensure_chartqa_vf_full
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan
echo "DeepSpeed OPD: ACCELERATE_CONFIG=${ACCELERATE_CONFIG} (ZeRO-0 default; override for ZeRO-2/3)"
echo "Teacher placement: DYME_TEACHER_DEVICE_MAP=${DYME_TEACHER_DEVICE_MAP} (auto colocates under DeepSpeed)"
echo "SFT cold-start / warmup: defaults in config/config_rlsd_chartqa.py (sft_cold_start_frac=0.08, etc.)"
echo "  Optional overrides: DYME_SFT_COLD_START_STEPS, DYME_SFT_COLD_START_FRAC, DYME_MAX_STEPS"

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
