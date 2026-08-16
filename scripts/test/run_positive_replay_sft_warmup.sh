#!/usr/bin/env bash
# Offline SFT warmup on positive replay targets, then use final_checkpoint for DyME.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${TEST_DIR}/../.." && pwd)"
cd "${ROOT}"

source "${ROOT}/scripts/test/launch_utils.sh"

DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  bash scripts/test/run_positive_replay_sft_warmup.sh [--dry-run]

Environment:
  This legacy convenience launcher was removed with Python/env configuration.
  Create a complete YAML recipe that explicitly sets dataset.train_dataset,
  training.sft_args.output_dir, and model.pretrained_model_path, then run
  main_sft.py with that YAML. Feed its final_checkpoint into an opd_only YAML.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Positive replay SFT launcher has been retired: create an explicit YAML recipe."
exit 2
