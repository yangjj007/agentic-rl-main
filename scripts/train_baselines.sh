#!/usr/bin/env bash
# Legacy launcher retained for backwards discoverability only.  Routing modes
# and model/data/loss settings must be selected by an explicit YAML recipe.
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/launch_utils.sh"

echo "scripts/train_baselines.sh is retired. Use an explicit YAML recipe:"
echo "  accelerate launch main.py --config config/config_trimode.yaml --mode rl"
echo "For post-SFT OPD, use config/config_opd_only_7b_chartqa.yaml."
exit 2
