"""Tests for OPSD hang-debug logging switches."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_debug_log_module():
    spec = importlib.util.spec_from_file_location(
        "opsd_debug_log_for_test",
        ROOT / "opsd_utils" / "debug_log.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_force_hang_probe_can_be_disabled(capsys):
    opsd_debug = _load_debug_log_module()
    opsd_debug.configure(rank=0, world_size=1, hang_force=False)

    opsd_debug.hang_probe_force("forced_probe")

    assert "OPSD-HANGDBG" not in capsys.readouterr().out


def test_force_hang_probe_logs_by_default(capsys):
    opsd_debug = _load_debug_log_module()
    opsd_debug.configure(rank=0, world_size=1, hang_force=True)

    opsd_debug.hang_probe_force("forced_probe")

    assert "OPSD-HANGDBG" in capsys.readouterr().out
