"""
Fast offline SFT baseline on full ChartQA with fewer epochs.

Usage (from test/):
  bash train_sft.sh
  accelerate launch ../main_sft.py --config test/config_rlsd_chartqa.py
"""
import os
import sys

_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

import config.config_rlsd_chartqa as base
from fast_profile import OUTPUT_ROOT, apply_to_config

OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "sft-rlsd-ref")

CONFIG = apply_to_config(
    base.CONFIG,
    output_dir=OUTPUT_DIR,
)

# Offline SFT writes to a dedicated subdir (not the RL output_dir above).
CONFIG["training"]["sft_args"]["output_dir"] = os.path.join(OUTPUT_ROOT, "sft")
