#!/usr/bin/env bash
# DyME / TriMode / OPSD ablation launchers (set MODE env var).
# Training params live in the config file selected per MODE.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

MODE="${MODE:-dyme}"
DYME_CONFIG="${DYME_CONFIG:-norm}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_accelerate_config)}"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

case "${MODE}" in
  dyme)
    accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
      --config "${DYME_CONFIG}" --mode rl
    ;;
  trimode|replace_sft|opsd_only|opsd_on_wrong|grpo_opsd_joint)
    DYME_CONFIG="${DYME_CONFIG:-trimode}"
    prepare_chartqa_training_data "${DYME_CONFIG}"
    accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
      --config "${DYME_CONFIG}" --mode rl --opsd_enabled --opsd_mode "${MODE}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use: dyme|trimode|replace_sft|opsd_only|opsd_on_wrong|grpo_opsd_joint"
    exit 1
    ;;
esac
