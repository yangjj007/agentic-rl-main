#!/usr/bin/env bash
# TriMode OPSD + DyME on ChartQA (LLaVA-OV 0.5B)
set -euo pipefail

cd "$(dirname "$0")/.."

export DYME_OPSD_MODE="${DYME_OPSD_MODE:-trimode}"
export DYME_OPSD_PROVIDERS="${DYME_OPSD_PROVIDERS:-text,visual_facts}"
export DYME_OUTPUT_DIR="${DYME_OUTPUT_DIR:-./outputs/trimode-chartqa}"

accelerate launch --config_file default_config.yaml main.py \
  --config config/config_trimode.py \
  --mode rl \
  --opsd_enabled \
  --opsd_mode "${DYME_OPSD_MODE}" \
  --opsd_providers "${DYME_OPSD_PROVIDERS}"
