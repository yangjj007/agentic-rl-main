#!/usr/bin/env bash
# Shared helpers for test/ fast baseline launch scripts.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${TEST_DIR}/.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/launch_utils.sh"

export DYME_DEPLOT_ENABLED="${DYME_DEPLOT_ENABLED:-0}"
export DYME_LOG_DIR="${DYME_LOG_DIR:-${ROOT}/outputs/test-fast/logs}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

prepare_fast_test_data() {
  bash "${TEST_DIR}/prepare_fast_dataset.sh"
}

fast_train_log_path() {
  local prefix="${1:-train}"
  train_log_path "${prefix}"
}

print_fast_plan() {
  local baseline="${1:-}"
  local config_path="${2:-}"
  local max_samples="${DYME_FAST_MAX_SAMPLES:-512}"
  local max_steps="${DYME_FAST_MAX_STEPS:-500}"
  local cold_frac="${DYME_FAST_COLD_START_FRAC:-0.08}"
  local cold_steps
  cold_steps="$(python - <<PY
frac = float("${cold_frac}")
steps = int("${max_steps}")
print(max(1, int(steps * frac)) if frac > 0 and steps > 0 else 0)
PY
)"
  echo "============================================================"
  echo "test/ fast baseline: ${baseline}"
  echo "config: ${config_path}"
  echo "samples: ${max_samples}  max_steps: ${max_steps}"
  if [[ "${baseline}" == "opd" ]]; then
    echo "OPD cold-start (embedded SFT): ${cold_steps}/${max_steps} steps"
    echo "OPD RL phase after cold start: $((max_steps - cold_steps)) steps"
  fi
  echo "output root: outputs/test-fast/"
  print_launch_plan
}
