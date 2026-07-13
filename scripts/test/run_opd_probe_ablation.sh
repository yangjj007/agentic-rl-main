#!/usr/bin/env bash
# Run the minimal teacher-probe ablation:
#   no-gold probe OPD vs no-probe OPD
#
# Default mode is dry-run so this script cannot accidentally disturb active GPU
# jobs. Use --run only after checking nvidia-smi.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -x "/home/deepseek_VG/.conda/envs/dyme/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/home/deepseek_VG/.conda/envs/dyme/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

MODE="dry-run"
MAX_STEPS="${DYME_ABLATION_MAX_STEPS:-500}"
RUN_ID="${DYME_ABLATION_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VARIANTS="no-gold,no-probe"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/test/run_opd_probe_ablation.sh [--dry-run|--run] [--max-steps N] [--variants no-gold,no-probe]

Examples:
  # Print exact commands only; does not start training.
  bash scripts/test/run_opd_probe_ablation.sh --dry-run

  # Run the 500-step ablation when GPUs are free.
  bash scripts/test/run_opd_probe_ablation.sh --run --max-steps 500

  # Run only the no-probe side.
  bash scripts/test/run_opd_probe_ablation.sh --run --variants no-probe --max-steps 500
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --variants)
      VARIANTS="$2"
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

case "${MAX_STEPS}" in
  ''|*[!0-9]*)
    echo "--max-steps must be a positive integer, got: ${MAX_STEPS}" >&2
    exit 2
    ;;
esac

if [[ "${MAX_STEPS}" -lt 1 ]]; then
  echo "--max-steps must be >= 1, got: ${MAX_STEPS}" >&2
  exit 2
fi

LOG_ROOT="${DYME_ABLATION_LOG_ROOT:-outputs/test-fast/logs/opd_probe_ablation_${RUN_ID}}"
OUT_ROOT="${DYME_ABLATION_OUTPUT_ROOT:-outputs/test-fast/opd-probe-ablation/${RUN_ID}}"

print_gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "============================================================"
    echo "GPU snapshot before ablation"
    local gpu_snapshot
    if ! gpu_snapshot="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1)"; then
      echo "WARN: nvidia-smi GPU snapshot unavailable in this shell."
      echo "============================================================"
      return 0
    fi
    echo "${gpu_snapshot}"
    echo "Active compute apps:"
    local app_snapshot
    if app_snapshot="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)"; then
      echo "${app_snapshot}"
    else
      echo "WARN: nvidia-smi compute-app snapshot unavailable in this shell."
    fi
    echo "============================================================"
  fi
}

variant_probe_flag() {
  local variant="$1"
  case "${variant}" in
    no-gold) echo "1" ;;
    no-probe) echo "0" ;;
    *)
      echo "Unknown variant: ${variant}" >&2
      return 2
      ;;
  esac
}

print_variant_command() {
  local variant="$1"
  local probe_flag="$2"
  local out_dir="${OUT_ROOT}/${variant}"
  local log_dir="${LOG_ROOT}/${variant}"

  cat <<CMD
DYME_TRAIN_MAX_STEPS=${MAX_STEPS} \\
DYME_MAX_STEPS=${MAX_STEPS} \\
DYME_OUTPUT_DIR=${out_dir} \\
DYME_LOG_DIR=${log_dir} \\
DYME_TEACHER_PROBE=${probe_flag} \\
DYME_TEACHER_TRAJECTORY=0 \\
DYME_VISUAL_CHECKER=0 \\
DYME_VISUAL_REFINER=0 \\
DYME_VISUAL_PREFETCH_IC=0 \\
DYME_VISUAL_LOG=0 \\
DYME_DEPLOT_ENABLED=0 \\
DYME_OPSD_HANG_DEBUG=0 \\
DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot \\
WANDB_MODE=disabled \\
bash scripts/train_opd_7b_dyme_probe.sh
CMD
}

check_log_has_paper_metrics() {
  local log_file="$1"
  local variant="$2"
  "${PYTHON_BIN}" - "$log_file" "$variant" <<'PY'
import ast
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
variant = sys.argv[2]
rows = []
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"\{.*\}", line)
    if not match:
        continue
    try:
        row = ast.literal_eval(match.group())
    except (SyntaxError, ValueError):
        continue
    if isinstance(row, dict) and any(k.startswith(("rewards/", "routing/", "completions/", "loss/")) for k in row):
        rows.append(row)

required = [
    "rewards/accuracy/mean",
    "rewards/format/mean",
    "completions/degenerate_rate",
    "completions/clipped_ratio",
    "routing/sft_replaced_ratio",
    "routing/grpo_on_correct_rate",
    "routing/opd_teacher_call_rate",
    "routing/teacher_probe_candidate_rate",
    "routing/teacher_probe_correct_rate",
    "routing/teacher_probe_wrong_rate",
    "loss/opsd",
]
missing = [key for key in required if not any(key in row for row in rows)]
if missing:
    raise SystemExit(f"{variant}: missing required metrics in {log_path}: {missing}")

def mean(key, window=None):
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if window:
        values = values[-window:]
    return sum(values) / len(values) if values else 0.0

print(f"### {variant} metric check: {log_path}")
print(f"metric_rows={len(rows)}")
for key in required:
    print(f"{key}: mean={mean(key):.4f} last20={mean(key, 20):.4f}")
PY
}

run_variant() {
  local variant="$1"
  local probe_flag
  probe_flag="$(variant_probe_flag "${variant}")"
  local out_dir="${OUT_ROOT}/${variant}"
  local log_dir="${LOG_ROOT}/${variant}"

  echo ""
  echo "============================================================"
  echo "Variant: ${variant}"
  echo "teacher probe enabled: ${probe_flag}"
  echo "max steps: ${MAX_STEPS}"
  echo "output dir: ${out_dir}"
  echo "log dir: ${log_dir}"
  echo "============================================================"

  if [[ "${MODE}" == "dry-run" ]]; then
    print_variant_command "${variant}" "${probe_flag}"
    return 0
  fi

  mkdir -p "${log_dir}" "${out_dir}"
  DYME_TRAIN_MAX_STEPS="${MAX_STEPS}" \
  DYME_MAX_STEPS="${MAX_STEPS}" \
  DYME_OUTPUT_DIR="${out_dir}" \
  DYME_LOG_DIR="${log_dir}" \
  DYME_TEACHER_PROBE="${probe_flag}" \
  DYME_TEACHER_TRAJECTORY=0 \
  DYME_VISUAL_CHECKER=0 \
  DYME_VISUAL_REFINER=0 \
  DYME_VISUAL_PREFETCH_IC=0 \
  DYME_VISUAL_LOG=0 \
  DYME_DEPLOT_ENABLED=0 \
  DYME_OPSD_HANG_DEBUG=0 \
  DYME_TEACHER_PROBE_PROVIDERS=format_only,visual_facts_deplot \
  WANDB_MODE=disabled \
  bash scripts/train_opd_7b_dyme_probe.sh

  local latest_log
  latest_log="$(ls -t "${log_dir}"/train_opd_7b_dyme_probe_*.log | head -1)"
  check_log_has_paper_metrics "${latest_log}" "${variant}"
  "${PYTHON_BIN}" scripts/analyze_opd_routes.py "${latest_log}" --window 20
}

echo "OPD teacher-probe ablation runner"
echo "mode: ${MODE}"
echo "run id: ${RUN_ID}"
echo "max steps: ${MAX_STEPS}"
echo "variants: ${VARIANTS}"
echo "log root: ${LOG_ROOT}"
echo "output root: ${OUT_ROOT}"
print_gpu_snapshot

IFS=',' read -ra _VARIANT_LIST <<< "${VARIANTS}"
for variant in "${_VARIANT_LIST[@]}"; do
  variant="${variant// /}"
  [[ -n "${variant}" ]] || continue
  run_variant "${variant}"
done

if [[ "${MODE}" == "dry-run" ]]; then
  echo ""
  echo "Dry-run only. Add --run when GPUs are free."
else
  echo ""
  echo "Ablation finished. Logs: ${LOG_ROOT}"
fi
