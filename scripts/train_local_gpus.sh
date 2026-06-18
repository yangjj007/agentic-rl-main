#!/usr/bin/env bash
# TriMode OPSD training on all GPUs visible to this machine.
# Training params: config/config_trimode_antidegen.py (default)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

DYME_CONFIG="${DYME_CONFIG:-config/config_trimode_antidegen.py}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"

NUM_PROCESSES="$(detect_num_gpus)"
if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No GPUs detected. Set NUM_GPUS=<n> or check CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

# Antidegen TriMode expects real DePlot visual facts by default.
export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-1}"
prepare_chartqa_training_data "${DYME_CONFIG}"

LOG_FILE="$(train_log_path "train_trimode_${NUM_PROCESSES}gpu")"
print_launch_plan
echo "Config: ${DYME_CONFIG}, log=${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  2>&1 | tee "${LOG_FILE}"
