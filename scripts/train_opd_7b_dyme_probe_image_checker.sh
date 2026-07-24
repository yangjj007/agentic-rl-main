#!/usr/bin/env bash
# Current best DyME teacher-probe OPD plus explicit image-primary visual checker.
#
# Usage:
#   bash scripts/train_opd_7b_dyme_probe_image_checker.sh
#   DYME_TRAIN_MAX_STEPS=5 bash scripts/train_opd_7b_dyme_probe_image_checker.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/test/launch_utils.sh"

DYME_CONFIG="opd_7b_dyme_probe_image_checker"
export DYME_VISUAL_CHECKER="${DYME_VISUAL_CHECKER:-1}"
export DYME_VISUAL_CHECKER_GROUNDING="${DYME_VISUAL_CHECKER_GROUNDING:-image_primary}"
export DYME_VISUAL_CHECKER_AUX="${DYME_VISUAL_CHECKER_AUX:-none}"

# Full training profile — do not inherit smoke profile env from the shell.
unset DYME_MAX_STEPS
if [[ -n "${DYME_TRAIN_MAX_STEPS:-}" ]]; then
  export DYME_MAX_STEPS="${DYME_TRAIN_MAX_STEPS}"
fi

export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$(resolve_deepspeed_zero1_config)}"
export DYME_OPSD_HANG_DEBUG="${DYME_OPSD_HANG_DEBUG:-0}"
export DYME_OPSD_HANG_FORCE="${DYME_OPSD_HANG_FORCE:-0}"

prepare_chartqa_training_data "${DYME_CONFIG}"
ensure_spacy_model

NUM_PROCESSES="$(detect_num_gpus)"
print_launch_plan

_TRAIN_PLAN="$("${PYTHON_BIN}" - <<'PY'
from config.loader import load_config

c = load_config("opd_7b_dyme_probe_image_checker")
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
echo "Config: ${DYME_CONFIG} (${_TRAIN_PLAN})"
echo "Snapshot: see output_dir/run_config_snapshot.json after launch"

LOG_FILE="$(train_log_path train_opd_7b_dyme_probe_image_checker)"
run_train_with_log "${LOG_FILE}" \
  "${PYTHON_BIN}" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG}" --num_processes "${NUM_PROCESSES}" main.py \
    --config "${DYME_CONFIG}" \
    --mode rl \
    --opsd_enabled \
    "$@"
