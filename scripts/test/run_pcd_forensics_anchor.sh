#!/usr/bin/env bash
# Run the missing no-PCD anchor for the PCD low-score forensic comparison.
#
# This intentionally trains only deplot_no_vs_opd. It does not launch any PCD
# batch-size repeats.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_ID="${DYME_PCD_FORENSICS_ANCHOR_RUN_ID:-pcd_forensics_anchor_4epoch}"
EPOCHS="${DYME_PCD_FORENSICS_ANCHOR_EPOCHS:-4}"
OUT_ROOT="${DYME_PCD_FORENSICS_ANCHOR_OUTPUT_ROOT:-outputs/test-fast/pcd-low-score-forensics/anchor_runs/${RUN_ID}}"
LOG_ROOT="${DYME_PCD_FORENSICS_ANCHOR_LOG_ROOT:-outputs/test-fast/pcd-low-score-forensics/logs/anchor_${RUN_ID}}"

DYME_DEPLOT_ABLATION_OUTPUT_ROOT="${OUT_ROOT}" \
DYME_DEPLOT_ABLATION_LOG_ROOT="${LOG_ROOT}" \
  bash scripts/test/run_opd_deplot_ablation.sh \
    --run \
    --run-id "${RUN_ID}" \
    --epochs "${EPOCHS}" \
    --variants deplot_no_vs_opd
