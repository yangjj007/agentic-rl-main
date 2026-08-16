#!/usr/bin/env bash
# DyME teacher-probe OPD on ChartQA — training params in config/config_opd_7b_dyme_probe.yaml
#
# Usage:
#   bash scripts/train_opd_7b_dyme_probe.sh
#   bash scripts/train_opd_7b_dyme_probe_smoke.sh   # 200-step short validation
#
# Memory-tight fallback:
#   ACCELERATE_CONFIG=default_config_zero2.yaml bash scripts/train_opd_7b_dyme_probe.sh
#
# Optional overrides (same command line only):
#   Copy config/config_opd_7b_dyme_probe.yaml, edit its explicit limits, then
#   pass that new YAML directly to main.py.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/test/launch_utils.sh"

# Full training profile — do not inherit smoke profile env from the shell.
CONFIG_PATH="config/config_opd_7b_dyme_probe.yaml"

export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"
prepare_chartqa_training_data "${CONFIG_PATH}"
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

_TRAIN_PLAN="$("${PYTHON_BIN}" - <<'PY'
from config.loader import load_config
c = load_config("config/config_opd_7b_dyme_probe.yaml")
args = c["training"]["dyme_args"]
max_steps = args.get("max_steps")
epochs = args.get("num_train_epochs")
if max_steps:
    print(f"max_steps={max_steps}")
else:
    print(f"num_train_epochs={epochs} (no max_steps cap)")
PY
)"
echo "Config: ${CONFIG_PATH} (${_TRAIN_PLAN})"
echo "Snapshot: see output_dir/run_config_snapshot.json after launch"

LOG_FILE="$(train_log_path train_opd_7b_dyme_probe)"
run_train_with_log "${LOG_FILE}" \
  "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
    --config "${CONFIG_PATH}" \
    --mode rl \
    --opsd_enabled \
    "$@"
