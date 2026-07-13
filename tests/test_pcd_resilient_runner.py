from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/test/run_pcd_no_visual_resilient.sh"
VARIANT = "deplot_no_vs_opd_pcd_oracle_hint_opd_no_full_hint_hard_sft_adaptive_supervision"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _base_env(tmp_path: Path, runner: Path, fake_bin: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DYME_PCD_RUNNER": str(runner),
        "DYME_PCD_RUN_ID": "pytest_resilient",
        "DYME_PCD_OUTPUT_ROOT": str(tmp_path / "out"),
        "DYME_PCD_LOG_ROOT": str(tmp_path / "logs"),
        "DYME_RESILIENT_STATE_DIR": str(tmp_path / "state"),
        "DYME_GPU_POLL_SECONDS": "0",
        "DYME_GPU_STABLE_SAMPLES": "1",
        "DYME_RETRY_WAIT_SECONDS": "0",
        "DYME_MAX_RETRIES": "2",
    }


def test_resilient_runner_waits_for_temperature_gate(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    query_count = tmp_path / "query_count"
    runner_called = tmp_path / "runner_called"
    runner = tmp_path / "runner.sh"
    _write_executable(
        fake_bin / "nvidia-smi",
        f"""#!/usr/bin/env bash
if [[ "$*" == *"--query-compute-apps="* ]]; then exit 0; fi
count=$(cat {query_count} 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > {query_count}
    temp=60
    util=100
    if [[ "$count" -ge 2 ]]; then util=0; fi
    for i in {{0..7}}; do echo "$i, 1000, $temp, $util"; done
""",
    )
    _write_executable(runner, f"#!/usr/bin/env bash\ntouch {runner_called}\n")

    result = subprocess.run(
        ["bash", str(SCRIPT), "4", "--variant", VARIANT],
        cwd=ROOT,
        env=_base_env(tmp_path, runner, fake_bin),
        text=True,
        capture_output=True,
        timeout=3,
    )

    assert result.returncode == 0, result.stderr
    assert query_count.read_text().strip() == "2"
    assert runner_called.exists()
    assert "util_pct=100" in result.stdout


def test_resilient_runner_waits_until_compute_processes_exit(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    process_query_count = tmp_path / "process_query_count"
    runner_called = tmp_path / "runner_called"
    runner = tmp_path / "runner.sh"
    _write_executable(
        fake_bin / "nvidia-smi",
        f"""#!/usr/bin/env bash
if [[ "$*" == *"--query-compute-apps="* ]]; then
  count=$(cat {process_query_count} 2>/dev/null || echo 0)
  count=$((count + 1))
  echo "$count" > {process_query_count}
  if [[ "$count" -eq 1 ]]; then
    echo "GPU-deadbeef, 4242, /other/job/python, 5134"
  fi
  exit 0
fi
for i in {{0..7}}; do echo "$i, 1000, 60, 0"; done
""",
    )
    _write_executable(runner, f"#!/usr/bin/env bash\ntouch {runner_called}\n")

    result = subprocess.run(
        ["bash", str(SCRIPT), "4", "--variant", VARIANT],
        cwd=ROOT,
        env=_base_env(tmp_path, runner, fake_bin),
        text=True,
        capture_output=True,
        timeout=3,
    )

    assert result.returncode == 0, result.stderr
    assert process_query_count.read_text().strip() == "2"
    assert runner_called.exists()
    assert "compute process" in result.stdout
    assert "pid=4242" in result.stdout


def test_resilient_runner_resumes_latest_checkpoint_after_transient_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls"
    runner = tmp_path / "runner.sh"
    out_dir = tmp_path / "out" / VARIANT
    _write_executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\nif [[ \"$*\" == *\"--query-compute-apps=\"* ]]; then exit 0; fi\nfor i in {0..7}; do echo \"$i, 1000, 60, 0\"; done\n",
    )
    _write_executable(
        runner,
        f"""#!/usr/bin/env bash
printf '%s\n' "$*" >> {calls}
count=$(wc -l < {calls})
if [[ "$count" -eq 1 ]]; then
  mkdir -p {out_dir}/checkpoint-50
  echo 'RuntimeError: CUDA error: unspecified launch failure'
  exit 1
fi
[[ "$*" == *'--resume auto'* ]]
""",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "4", "--variant", VARIANT],
        cwd=ROOT,
        env=_base_env(tmp_path, runner, fake_bin),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text().splitlines()
    assert len(call_lines) == 2
    assert "--resume none" in call_lines[0]
    assert "--resume auto" in call_lines[1]
    assert "transient failure detected" in result.stdout


def test_resilient_runner_does_not_retry_non_transient_failure(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls"
    runner = tmp_path / "runner.sh"
    _write_executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\nif [[ \"$*\" == *\"--query-compute-apps=\"* ]]; then exit 0; fi\nfor i in {0..7}; do echo \"$i, 1000, 60, 0\"; done\n",
    )
    _write_executable(
        runner,
        f"#!/usr/bin/env bash\necho call >> {calls}\necho 'ValueError: invalid config'\nexit 1\n",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "4", "--variant", VARIANT],
        cwd=ROOT,
        env=_base_env(tmp_path, runner, fake_bin),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert calls.read_text().splitlines() == ["call"]
    assert "non-transient failure" in result.stdout
