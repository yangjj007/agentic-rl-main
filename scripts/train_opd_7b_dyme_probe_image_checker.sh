#!/usr/bin/env bash
# Current best DyME teacher-probe OPD plus explicit image-primary visual checker.
#
# Usage:
#   bash scripts/train_opd_7b_dyme_probe_image_checker.sh
#   Copy config/config_opd_7b_dyme_probe_image_checker.yaml, edit its explicit
#   limits, then pass that new YAML directly to main.py.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

CONFIG_PATH="config/config_opd_7b_dyme_probe_image_checker.yaml"

export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"
prepare_chartqa_training_data "${CONFIG_PATH}"

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

_TRAIN_PLAN="$("${PYTHON_BIN}" - <<'PY'
from config.loader import load_config

c = load_config("config/config_opd_7b_dyme_probe_image_checker.yaml")
args = c["training"]["dyme_args"]
visual = c["opsd"]["visual_supervision"]
max_steps = args.get("max_steps")
epochs = args.get("num_train_epochs")
if max_steps:
    horizon = f"max_steps={max_steps}"
else:
    horizon = f"num_train_epochs={epochs} (no max_steps cap)"
checker = visual.get("checker", {})
print(
    f"{horizon}; checker={checker.get('enabled')} "
    f"grounding={checker.get('grounding')} aux={checker.get('aux_evidence')} "
    f"refiner={visual.get('refiner', {}).get('enabled')} "
    f"prefetch_ic={visual.get('prefetch_ic')}"
)
PY
)"
echo "Config: ${CONFIG_PATH} (${_TRAIN_PLAN})"
echo "Snapshot: see output_dir/run_config_snapshot.json after launch"

LOG_FILE="$(train_log_path train_opd_7b_dyme_probe_image_checker)"
run_train_with_log "${LOG_FILE}" \
  "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
    --config "${CONFIG_PATH}" \
    --mode rl \
    --opsd_enabled \
    "$@"
