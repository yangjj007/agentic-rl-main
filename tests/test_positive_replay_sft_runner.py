from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_positive_replay_sft_warmup_requires_an_explicit_yaml_recipe() -> None:
    result = subprocess.run(
        ["bash", "scripts/test/run_positive_replay_sft_warmup.sh", "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "explicit YAML recipe" in result.stdout
