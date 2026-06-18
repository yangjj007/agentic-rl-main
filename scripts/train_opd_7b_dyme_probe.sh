#!/usr/bin/env bash
# DyME-aligned teacher-probe OPD on ChartQA (dyme_teacher_probe_opd).
#
# Routing (config/config_opd_7b_dyme_probe.py):
#   all-wrong group  -> SFT (every completion)
#   correct          -> GRPO
#   wrong            -> 7B teacher answer probe
#   teacher-correct  -> SRKL OPD (+ optional teacher-trajectory FKL)
#   teacher-wrong    -> SFT
#
# Teacher privileged context (no gold): format_only + visual_facts by default.
# Requires data/chartqa/train_medium_vf_full.json (auto-built by ensure_chartqa_vf_full).
#
# Usage:
#   bash scripts/train_opd_7b_dyme_probe.sh
#
# Memory-tight fallback:
#   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_dyme_probe.sh
#   ACCELERATE_CONFIG=default_config_zero3_colocate.yaml bash scripts/train_opd_7b_dyme_probe.sh
#
# Ablation examples:
#   DYME_TEACHER_PROBE=0 bash scripts/train_opd_7b_dyme_probe.sh
#   DYME_TEACHER_PROBE_PROVIDERS=format_only bash scripts/train_opd_7b_dyme_probe.sh
#   DYME_OPSD_LOSS_TYPE=jsd DYME_TEACHER_TRAJECTORY=0 bash scripts/train_opd_7b_dyme_probe.sh
#
# Local models (use absolute paths):
#   export DYME_STUDENT_MODEL=/path/to/llava-0.5b-ov
#   export DYME_TEACHER_MODEL=/path/to/llava-7b-ov
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

# --- DyME teacher-probe routing (config defaults match these env vars) ---
export DYME_OPSD_MODE="${DYME_OPSD_MODE:-dyme_teacher_probe_opd}"
export DYME_OPSD_PRIVILEGE_PROFILE="${DYME_OPSD_PRIVILEGE_PROFILE:-hybrid}"
export DYME_TEACHER_PROBE_PROVIDERS="${DYME_TEACHER_PROBE_PROVIDERS:-format_only,visual_facts}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-${DYME_TEACHER_PROBE_PROVIDERS}}"

export DYME_TEACHER_PROBE="${DYME_TEACHER_PROBE:-1}"
export DYME_TEACHER_TRAJECTORY="${DYME_TEACHER_TRAJECTORY:-1}"
export DYME_VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-1}"
export DYME_VISUAL_REFINER="${DYME_VISUAL_REFINER:-1}"

export DYME_OPSD_LOSS_TYPE="${DYME_OPSD_LOSS_TYPE:-srkl}"
export DYME_OPSD_SRKL_ALPHA="${DYME_OPSD_SRKL_ALPHA:-0.1}"
export DYME_TEACHER_TRAJ_FKL_WEIGHT="${DYME_TEACHER_TRAJ_FKL_WEIGHT:-0.5}"

export DYME_TEACHER_MODEL="${DYME_TEACHER_MODEL:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/opd-7b-dyme-probe-chartqa}"

# ZeRO-0 default (8 GPU -> default_config_8gpu_deepspeed.yaml; else default_config_deepspeed.yaml)
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero0_config)}"
export DYME_TEACHER_DEVICE_MAP="${DYME_TEACHER_DEVICE_MAP:-auto}"
export DYME_OPSD_DETAIL_MIN_FREE_GB="${DYME_OPSD_DETAIL_MIN_FREE_GB:-4.0}"
export DYME_OPSD_DETAIL_EVERY="${DYME_OPSD_DETAIL_EVERY:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DYME_GRADIENT_CHECKPOINTING="${DYME_GRADIENT_CHECKPOINTING:-0}"

# visual_facts needs enriched JSON; DePlot placeholder is fine for a quick start (DYME_DEPLOT_ENABLED=1 for real DePlot)
export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-0}"
ensure_chartqa_vf_full
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

echo "DyME teacher-probe OPD: mode=${DYME_OPSD_MODE}"
echo "  privileged providers: ${DYME_TEACHER_PROBE_PROVIDERS}"
echo "  teacher probe: ${DYME_TEACHER_PROBE} | trajectory FKL: ${DYME_TEACHER_TRAJECTORY}"
echo "  loss: ${DYME_OPSD_LOSS_TYPE} (srkl_alpha=${DYME_OPSD_SRKL_ALPHA})"
echo "  visual checker/refiner: ${DYME_VISUAL_CHECKER}/${DYME_VISUAL_REFINER}"
echo "  output: ${DYME_OUTPUT_DIR}"
echo "  ACCELERATE_CONFIG=${ACCELERATE_CONFIG} (override for ZeRO-2/3 if OOM)"
echo "  config snapshot -> ${DYME_OUTPUT_DIR}/run_config_snapshot.json"

LOG_DIR="${DYME_LOG_DIR:-./outputs/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_opd_7b_dyme_probe_$(date +%Y%m%d_%H%M%S).log"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config opd_7b_dyme_probe \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}" \
  --opsd_privilege_profile "${DYME_OPSD_PRIVILEGE_PROFILE}" \
  2>&1 | tee "${LOG_FILE}"
