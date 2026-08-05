"""Lightweight checks for checkpoint-evaluation entrypoint wiring.

These deliberately extract the small pure helpers from ``main.py`` instead of
importing it: importing the training entrypoint requires CUDA/TRL packages,
which are not part of a unit-test environment.
"""
import ast
import __future__
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from opsd_utils.checkpoint_eval_paths import (
    recover_interrupted_checkpoint_eval_save,
    validate_checkpoint_eval_output_dir,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = ROOT / "main.py"


def _load_main_helper(name: str, extra_globals: dict[str, Any] | None = None):
    tree = ast.parse(MAIN_SOURCE.read_text(encoding="utf-8"))
    node = next(
        child for child in tree.body if isinstance(child, ast.FunctionDef) and child.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Any": Any, "Dataset": object}
    namespace.update(extra_globals or {})
    exec(
        compile(
            module,
            str(MAIN_SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace[name]


def _load_main_helpers(
    names: tuple[str, ...], extra_globals: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract a small cooperating helper group without importing CUDA/TRL main."""
    tree = ast.parse(MAIN_SOURCE.read_text(encoding="utf-8"))
    wanted = set(names)
    nodes = [
        child
        for child in tree.body
        if isinstance(child, ast.FunctionDef) and child.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"Any": Any, "Path": Path, "os": os}
    namespace.update(extra_globals or {})
    exec(
        compile(
            module,
            str(MAIN_SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return {name: namespace[name] for name in names}


def test_checkpoint_eval_split_selects_validation_and_never_test_fallback() -> None:
    select_split = _load_main_helper("_select_chartqa_eval_split")
    validation = object()
    test = object()

    assert select_split({"validation": validation, "test": test}, "validation") is validation
    # A common mirror spelling remains supported.
    val = object()
    assert select_split({"val": val, "test": test}, "validation") is val

    with pytest.raises(ValueError, match="Refusing to fall back to test"):
        select_split({"test": test}, "validation")
    with pytest.raises(ValueError, match="validation"):
        select_split({"validation": validation, "test": test}, "test")


def test_checkpoint_eval_config_rejects_test_for_training_model_selection() -> None:
    defaults = {
        "enabled": False,
        "split": "validation",
        "batch_size": 1,
        "max_new_tokens": 1024,
        "patience": 3,
        "tie_policy": "reset",
    }
    resolve = _load_main_helper(
        "resolve_checkpoint_eval_config", {"_CHECKPOINT_EVAL_DEFAULTS": defaults}
    )
    with pytest.raises(ValueError, match="validation or val"):
        resolve(
            {"enabled": True, "split": "test"},
            task="chartqa",
            eval_dataset=[1],
            dyme_args={"save_strategy": "steps", "output_dir": "/tmp/out"},
        )


def test_checkpoint_eval_config_uses_hf_steps_default_and_disables_generic_eval() -> None:
    defaults = {
        "enabled": False,
        "split": "validation",
        "batch_size": 1,
        "max_new_tokens": 1024,
        "patience": 3,
        "tie_policy": "reset",
    }
    resolve = _load_main_helper(
        "resolve_checkpoint_eval_config", {"_CHECKPOINT_EVAL_DEFAULTS": defaults}
    )
    dyme_args = {"output_dir": "/tmp/out", "save_steps": 100}

    resolved = resolve(
        {"enabled": True},
        task="chartqa",
        eval_dataset=[1],
        dyme_args=dyme_args,
    )

    assert resolved["enabled"] is True
    # Profiles that only set save_steps rely on TrainingArguments' default
    # save strategy.  Generic eval must nevertheless be disabled because its
    # metrics do not include the custom checkpoint score.
    assert dyme_args["eval_strategy"] == "no"
    assert dyme_args["save_total_limit"] == 1
    assert dyme_args["restore_callback_states_from_checkpoint"] is True


def test_smoke_configs_explicitly_disable_checkpoint_evaluation() -> None:
    for relative in (
        "config/config_opd_7b_smoke.py",
        "config/config_opd_7b_dyme_probe_smoke.py",
        "config/config_rlsd_shortrun.py",
        "scripts/test/config/config_opd_force_smoke.py",
        "scripts/test/config/config_positive_replay_sft.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '"enabled": False' in source, relative

    # The remaining short baseline wrappers share `fast_profile`; keep their
    # override centralized so full-validation checkpoint selection is never
    # accidentally enabled by an inherited production ChartQA config.
    fast_profile = (ROOT / "scripts/test/config/fast_profile.py").read_text(encoding="utf-8")
    assert 'cfg["checkpoint_eval"] = {**cfg["checkpoint_eval"], "enabled": False}' in fast_profile


def test_non_chartqa_configs_hard_disable_checkpoint_evaluation() -> None:
    for relative in ("config/config_aok.py", "config/config_llm.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '\"checkpoint_eval\": {\"enabled\": False}' in source, relative
        assert "DYME_CHECKPOINT_EVAL" not in source, relative

    # Import in a fresh interpreter so an inherited production environment
    # cannot accidentally opt these non-ChartQA tasks into ChartQA scoring.
    code = """
from config.loader import load_config
for name in (\"aok\", \"llm\"):
    print(load_config(name)[\"checkpoint_eval\"][\"enabled\"])
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "DYME_CHECKPOINT_EVAL": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["False", "False"]


def test_short_runner_commands_explicitly_disable_checkpoint_evaluation() -> None:
    image_checker = (ROOT / "scripts/test/run_image_checker_timing_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "export DYME_CHECKPOINT_EVAL=0" in image_checker

    probe_ablation = (ROOT / "scripts/test/run_opd_probe_ablation.sh").read_text(
        encoding="utf-8"
    )
    # Keep both copied dry-run commands and the actual launch environment safe.
    assert probe_ablation.count("DYME_CHECKPOINT_EVAL=0") == 2

    deplot_ablation = (ROOT / "scripts/test/run_opd_deplot_ablation.sh").read_text(
        encoding="utf-8"
    )
    # The full ablation retains its production behavior; only --smoke gets
    # the override, once for its displayed command and once at launch.
    assert deplot_ablation.count("DYME_CHECKPOINT_EVAL=0") == 2


def test_enabled_main_path_uses_best_link_instead_of_terminal_save() -> None:
    source = MAIN_SOURCE.read_text(encoding="utf-8")
    enabled_branch = source.split("if checkpoint_eval_config.get(\"enabled\"):", 1)[-1]
    assert "find_best_checkpoint_path" in enabled_branch
    assert "update_final_checkpoint_link" in enabled_branch


def test_recovery_rewrites_only_an_explicit_checkpoint_path_it_removed(tmp_path: Path) -> None:
    helpers = _load_main_helpers(
        (
            "_explicit_checkpoint_resume_target",
            "_rewrite_recovered_resume_checkpoint",
        )
    )
    old_checkpoint = tmp_path / "checkpoint-100"
    new_checkpoint = tmp_path / "checkpoint-200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    before_recovery = helpers["_explicit_checkpoint_resume_target"](
        str(old_checkpoint), output_dir=tmp_path
    )

    shutil.rmtree(old_checkpoint)
    rewritten = helpers["_rewrite_recovered_resume_checkpoint"](
        str(old_checkpoint),
        resume_target_before_recovery=before_recovery,
        recovered_checkpoint=new_checkpoint,
    )
    assert rewritten == str(new_checkpoint.resolve())

    # Missing or external paths were not proven to be the rotation victim and
    # keep their ordinary Trainer error semantics.
    missing = tmp_path / "checkpoint-300"
    assert helpers["_rewrite_recovered_resume_checkpoint"](
        str(missing),
        resume_target_before_recovery=None,
        recovered_checkpoint=new_checkpoint,
    ) == str(missing)
    assert helpers["_rewrite_recovered_resume_checkpoint"](
        str(new_checkpoint),
        resume_target_before_recovery=new_checkpoint.resolve(),
        recovered_checkpoint=new_checkpoint,
    ) == str(new_checkpoint)


def test_startup_recovery_repairs_final_checkpoint_before_its_explicit_resume(
    tmp_path: Path,
) -> None:
    """The post-rotation link repair must happen before Trainer sees the path."""
    old_checkpoint = tmp_path / "checkpoint-100"
    new_checkpoint = tmp_path / "checkpoint-200"
    new_checkpoint.mkdir()
    old_policy = {
        "version": 1,
        "patience": 3,
        "tie_policy": "reset",
        "state": {
            "best_score": 0.60,
            "best_step": 100,
            "lower_score_streak": 0,
            "evaluation_count": 1,
            "stop_requested": False,
        },
    }
    new_policy = {
        "version": 1,
        "patience": 3,
        "tie_policy": "reset",
        "state": {
            "best_score": 0.75,
            "best_step": 200,
            "lower_score_streak": 0,
            "evaluation_count": 2,
            "stop_requested": False,
        },
    }
    (new_checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 200,
                "best_global_step": 200,
                "stateful_callbacks": {
                    "CheckpointEvaluationTriggerCallback": {
                        "attributes": {"checkpoint_eval_state": new_policy}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_eval_state.json").write_text(
        json.dumps({"version": 1, "checkpoint_eval_state": old_policy}), encoding="utf-8"
    )
    # This is exactly the HF rotation -> callback-on_save crash state.
    os.symlink(old_checkpoint.name, tmp_path / "final_checkpoint")
    requested_resume = str(tmp_path / "final_checkpoint")

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> list[Any]:
        assert from_process == 0
        return objects

    helpers = _load_main_helpers(
        (
            "_explicit_checkpoint_resume_target",
            "_rewrite_recovered_resume_checkpoint",
            "_coordinate_checkpoint_eval_recovery",
        ),
        {
            "recover_interrupted_checkpoint_eval_save": recover_interrupted_checkpoint_eval_save,
            "validate_checkpoint_eval_output_dir": validate_checkpoint_eval_output_dir,
            "broadcast_object_list": fake_broadcast,
        },
    )

    class _MainAccelerator:
        is_main_process = True

    resume, recovered = helpers["_coordinate_checkpoint_eval_recovery"](
        accelerator=_MainAccelerator(),
        output_dir=tmp_path,
        patience=3,
        tie_policy="reset",
        resume_from_checkpoint=requested_resume,
    )

    # main intentionally preserves the user's final_checkpoint spelling; the
    # coordinated recovery has already made that same path valid for Trainer.
    assert resume == requested_resume
    assert recovered == str(new_checkpoint)
    assert Path(resume).resolve() == new_checkpoint.resolve()


def test_rank_zero_recovery_outcome_is_broadcast_before_workers_proceed(tmp_path: Path) -> None:
    old_checkpoint = tmp_path / "checkpoint-100"
    new_checkpoint = tmp_path / "checkpoint-200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    calls: list[tuple[str, Any]] = []

    def fake_recover(output_dir: str, *, patience: int, tie_policy: str) -> Path:
        calls.append(("recover", (output_dir, patience, tie_policy)))
        shutil.rmtree(old_checkpoint)
        return new_checkpoint

    def fake_validate(output_dir: str) -> None:
        calls.append(("validate", output_dir))

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> list[Any]:
        calls.append(("broadcast", (objects[0], from_process)))
        return objects

    helpers = _load_main_helpers(
        (
            "_explicit_checkpoint_resume_target",
            "_rewrite_recovered_resume_checkpoint",
            "_coordinate_checkpoint_eval_recovery",
        ),
        {
            "recover_interrupted_checkpoint_eval_save": fake_recover,
            "validate_checkpoint_eval_output_dir": fake_validate,
            "broadcast_object_list": fake_broadcast,
        },
    )

    class _MainAccelerator:
        is_main_process = True

    resume, recovered = helpers["_coordinate_checkpoint_eval_recovery"](
        accelerator=_MainAccelerator(),
        output_dir=tmp_path,
        patience=3,
        tie_policy="reset",
        resume_from_checkpoint=str(old_checkpoint),
    )

    assert resume == str(new_checkpoint.resolve())
    assert recovered == str(new_checkpoint)
    assert [name for name, _ in calls] == ["recover", "validate", "broadcast"]
    assert calls[-1][1][0]["ok"] is True


def test_rank_zero_recovery_error_is_broadcast_instead_of_waiting_at_a_barrier(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_recover(*args: Any, **kwargs: Any) -> None:
        calls.append(("recover", None))
        raise ValueError("ambiguous checkpoint layout")

    def fake_validate(*args: Any, **kwargs: Any) -> None:
        pytest.fail("validation must not run after recovery failed")

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> list[Any]:
        calls.append(("broadcast", (objects[0], from_process)))
        return objects

    helpers = _load_main_helpers(
        (
            "_explicit_checkpoint_resume_target",
            "_rewrite_recovered_resume_checkpoint",
            "_coordinate_checkpoint_eval_recovery",
        ),
        {
            "recover_interrupted_checkpoint_eval_save": fake_recover,
            "validate_checkpoint_eval_output_dir": fake_validate,
            "broadcast_object_list": fake_broadcast,
        },
    )

    class _MainAccelerator:
        is_main_process = True

    with pytest.raises(RuntimeError, match="rank-zero recovery/layout validation failed"):
        helpers["_coordinate_checkpoint_eval_recovery"](
            accelerator=_MainAccelerator(),
            output_dir=tmp_path,
            patience=3,
            tie_policy="reset",
            resume_from_checkpoint=None,
        )
    assert [name for name, _ in calls] == ["recover", "broadcast"]
    assert calls[-1][1][0] == {
        "ok": False,
        "error_type": "ValueError",
        "error": "ambiguous checkpoint layout",
    }


def test_worker_uses_rank_zero_recovered_resume_path_and_error(tmp_path: Path) -> None:
    new_checkpoint = tmp_path / "checkpoint-200"
    new_checkpoint.mkdir()
    transmitted: dict[str, Any] = {
        "ok": True,
        "recovered_checkpoint": str(new_checkpoint),
        "resume_from_checkpoint": str(new_checkpoint),
    }

    def fake_recover(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a worker must not attempt destructive recovery")

    def fake_validate(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a worker must use rank-zero layout validation")

    def fake_broadcast(objects: list[Any], from_process: int = 0) -> list[Any]:
        assert objects == [None]
        assert from_process == 0
        objects[0] = dict(transmitted)
        return objects

    helpers = _load_main_helpers(
        (
            "_explicit_checkpoint_resume_target",
            "_rewrite_recovered_resume_checkpoint",
            "_coordinate_checkpoint_eval_recovery",
        ),
        {
            "recover_interrupted_checkpoint_eval_save": fake_recover,
            "validate_checkpoint_eval_output_dir": fake_validate,
            "broadcast_object_list": fake_broadcast,
        },
    )

    class _WorkerAccelerator:
        is_main_process = False

    assert helpers["_coordinate_checkpoint_eval_recovery"](
        accelerator=_WorkerAccelerator(),
        output_dir=tmp_path,
        patience=3,
        tie_policy="reset",
        resume_from_checkpoint=str(tmp_path / "checkpoint-100"),
    ) == (str(new_checkpoint), str(new_checkpoint))

    transmitted.update(
        {
            "ok": False,
            "error_type": "ValueError",
            "error": "ambiguous checkpoint layout",
        }
    )
    with pytest.raises(RuntimeError, match="rank-zero recovery/layout validation failed"):
        helpers["_coordinate_checkpoint_eval_recovery"](
            accelerator=_WorkerAccelerator(),
            output_dir=tmp_path,
            patience=3,
            tie_policy="reset",
            resume_from_checkpoint=None,
        )
