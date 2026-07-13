from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_pcd_variant_smoke",
        ROOT / "scripts" / "analysis" / "check_pcd_variant_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_eval_format_smoke_check_requires_eval_format_metric(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text("{'reward/eval_format_mean': 0.75}\n>>> Training finished OK\n", encoding="utf-8")
    module = _load_module()

    rc = module.main(
        [
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_eval_format_reward",
            "--log-file",
            str(log),
        ]
    )

    assert rc == 0


def test_late_traj_smoke_check_reports_missing_effective_weight(tmp_path: Path, capsys) -> None:
    log = tmp_path / "train.log"
    log.write_text("{'loss/teacher_traj_fkl': 0.1}\n>>> Training finished OK\n", encoding="utf-8")
    module = _load_module()

    rc = module.main(
        [
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_late_traj_decay",
            "--log-file",
            str(log),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "loss/teacher_traj_effective_weight" in captured.out


def test_combo_smoke_check_requires_both_new_metrics(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "{'reward/eval_format_mean': 0.8, 'loss/teacher_traj_effective_weight': 0.5}\n"
        ">>> Training finished OK\n",
        encoding="utf-8",
    )
    module = _load_module()

    rc = module.main(
        [
            "--variant",
            "deplot_no_vs_opd_pcd_oracle_hint_eval_format_late_traj_decay",
            "--log-file",
            str(log),
        ]
    )

    assert rc == 0
