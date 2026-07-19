from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_refiner_sft_repair_tokenizes_explicit_teacher_probe_target() -> None:
    trainer = (ROOT / "trainer" / "DyMETrainer.py").read_text(encoding="utf-8")

    assert re.search(
        r'repair_cfg\["mode"\]\s+in\s+\([^)]*"traj_sft"[^)]*"refiner_sft"[^)]*\)',
        trainer,
    )


def test_refiner_sft_keeps_teacher_probe_text_without_teacher_trajectory() -> None:
    trainer = (ROOT / "trainer" / "DyMETrainer.py").read_text(encoding="utf-8")
    correct_probe_block = re.search(
        r"if score > 0:(?P<body>.*?)else:\s*\n\s*stats\[\"teacher_probe_wrong\"\]",
        trainer,
        re.S,
    )

    assert correct_probe_block is not None
    body = correct_probe_block.group("body")
    assert "teacher_traj_texts[global_idx] = text" in body
    assert body.index("teacher_traj_texts[global_idx] = text") < body.index(
        'if self._teacher_trajectory_config()["enabled"]'
    )


def test_online_sft_targets_use_configurable_formatter() -> None:
    trainer = (ROOT / "trainer" / "DyMETrainer.py").read_text(encoding="utf-8")

    assert "def _format_online_sft_target(" in trainer
    assert "online_sft_target_style" in trainer
    assert "hints[src] + \"\\n\" + answers[src] + self.end_flag" not in trainer
    assert "hint + '\\n' + answer + self.end_flag" not in trainer
