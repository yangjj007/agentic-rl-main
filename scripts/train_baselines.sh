#!/usr/bin/env bash
# DyME / TriMode / OPSD ablation launchers (set MODE env var)
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

MODE="${MODE:-dyme}"
CONFIG="${CONFIG:-config/config.py}"
PROVIDERS="${DYME_OPSD_PROVIDERS:-text}"
ACCELERATE_CONFIG="$(resolve_accelerate_config)"
NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

case "${MODE}" in
  dyme)
    accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py --config "${CONFIG}" --mode rl
    ;;
  trimode|replace_sft|opsd_only|opsd_on_wrong|grpo_opsd_joint)
    accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
      --config config/config_trimode.py --mode rl \
      --opsd_enabled --opsd_mode "${MODE}" --opsd_providers "${PROVIDERS}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use: dyme|trimode|replace_sft|opsd_only|opsd_on_wrong|grpo_opsd_joint"
    exit 1
    ;;
esac
