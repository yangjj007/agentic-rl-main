#!/usr/bin/env bash
# DyME teacher-probe OPD on ChartQA — training params in config/config_opd_7b_dyme_probe.py
#
# Usage:
#   bash scripts/train_opd_7b_dyme_probe.sh
#   bash scripts/train_opd_7b_dyme_probe_smoke.sh   # 200-step short validation
#
# Memory-tight fallback:
#   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_dyme_probe.sh
#
# Optional overrides (same command line only):
#   DYME_TRAIN_MAX_STEPS=500 DYME_NUM_TRAIN_EPOCHS=5 bash scripts/train_opd_7b_dyme_probe.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

# Full training profile — do not inherit smoke profile env from the shell.
DYME_CONFIG="opd_7b_dyme_probe"
unset DYME_MAX_STEPS
if [[ -n "${DYME_TRAIN_MAX_STEPS:-}" ]]; then
  export DYME_MAX_STEPS="${DYME_TRAIN_MAX_STEPS}"
fi

export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"

prepare_chartqa_training_data "${DYME_CONFIG}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

_TRAIN_PLAN="$(python - <<'PY'
from config.loader import load_config
c = load_config("opd_7b_dyme_probe")
args = c["training"]["dyme_args"]
max_steps = args.get("max_steps")
epochs = args.get("num_train_epochs")
if max_steps:
    print(f"max_steps={max_steps}")
else:
    print(f"num_train_epochs={epochs} (no max_steps cap)")
PY
)"
echo "Config: ${DYME_CONFIG} (${_TRAIN_PLAN})"
echo "Snapshot: see output_dir/run_config_snapshot.json after launch"

LOG_FILE="$(train_log_path train_opd_7b_dyme_probe)"
echo "Writing log to: ${LOG_FILE}"

accelerate launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
  --config "${DYME_CONFIG}" \
  --mode rl \
  --opsd_enabled \
  --no_wandb \
  2>&1 | tee "${LOG_FILE}"
