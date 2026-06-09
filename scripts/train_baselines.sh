#!/usr/bin/env bash
# DyME / TriMode / OPSD ablation launchers (set MODE env var)
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${MODE:-dyme}"
CONFIG="${CONFIG:-config/config.py}"
PROVIDERS="${DYME_OPSD_PROVIDERS:-text}"

case "${MODE}" in
  dyme)
    accelerate launch --config_file default_config.yaml main.py --config "${CONFIG}" --mode rl
    ;;
  trimode|replace_sft|opsd_only|opsd_on_wrong|grpo_opsd_joint)
    accelerate launch --config_file default_config.yaml main.py \
      --config config/config_trimode.py --mode rl \
      --opsd_enabled --opsd_mode "${MODE}" --opsd_providers "${PROVIDERS}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use: dyme|trimode|replace_sft|opsd_only|opsd_on_wrong|grpo_opsd_joint"
    exit 1
    ;;
esac
