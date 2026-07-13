#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
exec bash scripts/test/run_pcd_no_visual.sh 10 --resume auto "$@"
