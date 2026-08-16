#!/usr/bin/env bash
# Run all three fast baselines sequentially: SFT -> OPD -> DyME.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${TEST_DIR}/launch_utils.sh"

echo "============================================================"
echo "scripts/test/ fast baselines — sequential run"
echo "  dataset: full ChartQA (train_medium_vf_full.json)"
echo "  Hyperparameters: explicit in each YAML recipe"
echo "  order: SFT -> OPD -> DyME"
echo "  outputs:"
echo "    SFT  -> outputs/test-fast/sft/final_checkpoint"
echo "    DyME -> outputs/test-fast/dyme/"
echo "    OPD  -> outputs/test-fast/opd-7b-ds/"
echo "============================================================"

run_test_baseline "SFT" "${TEST_DIR}/train_sft.sh"
run_test_baseline "OPD" "${TEST_DIR}/train_opd.sh"
run_test_baseline "DyME" "${TEST_DIR}/train_dyme.sh"

echo "All fast baselines finished."
