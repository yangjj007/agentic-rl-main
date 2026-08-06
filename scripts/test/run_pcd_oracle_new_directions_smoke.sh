#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="${DYME_PCD_RUN_ID:-pcd_oracle_new_directions_smoke10}"
MAX_STEPS="${DYME_PCD_MAX_STEPS:-10}"
DRY_RUN=1
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_pcd_oracle_new_directions_smoke.sh [--dry-run|--run] [--run-id ID]

Runs the three oracle PCD new-direction smoke variants with DYME_PCD_MAX_STEPS=10
and checks each log for its required diagnostic metric.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --run)
      DRY_RUN=0
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

VARIANTS=(
  "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward"
  "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay"
  "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay"
)

echo "PCD oracle new-direction smoke"
echo "run id: ${RUN_ID}"
echo "DYME_PCD_MAX_STEPS=${MAX_STEPS}"
echo "mode: $([[ "${DRY_RUN}" == "1" ]] && echo dry-run || echo run)"

for variant in "${VARIANTS[@]}"; do
  log_dir="${DYME_PCD_LOG_ROOT:-outputs/test-fast/logs/pcd_no_visual_${RUN_ID}}/${variant}"
  echo "============================================================"
  echo "variant: ${variant}"
  train_cmd=(
    env
    "DYME_PCD_RUN_ID=${RUN_ID}"
    "DYME_PCD_MAX_STEPS=${MAX_STEPS}"
    bash scripts/test/run_pcd_no_visual_4epoch.sh
    --variant "${variant}"
  )
  check_cmd=(
    "${PYTHON_BIN}"
    scripts/analysis/check_pcd_variant_smoke.py
    --variant "${variant}"
    --log-dir "${log_dir}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'would run:'
    printf ' %q' "${train_cmd[@]}"
    printf '\n'
    printf 'would check:'
    printf ' %q' "${check_cmd[@]}"
    printf '\n'
  else
    "${train_cmd[@]}"
    "${check_cmd[@]}"
  fi
done
