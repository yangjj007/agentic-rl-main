"""Tests for local model path resolution."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.paths import (
    discover_local_model,
    local_pretrained_kwargs,
    resolve_model_path,
    validate_local_model_dir,
)


ROOT = Path(__file__).resolve().parents[1]


def test_hub_id_unchanged():
    assert resolve_model_path("llava-hf/llava-onevision-qwen2-0.5b-ov-hf") == (
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    )


def test_validate_local_requires_weights(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        validate_local_model_dir(str(empty), role="student")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "student" in str(exc)


def test_validate_missing_absolute_path_fails_early(tmp_path):
    missing = tmp_path / "missing-model"
    try:
        validate_local_model_dir(str(missing), role="student")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "student" in str(exc)
        assert "does not exist" in str(exc)


def test_discover_local_model_from_env(tmp_path, monkeypatch):
    model = tmp_path / "llava-0.5b-ov"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("DYME_STUDENT_MODEL", str(model))
    assert discover_local_model("student", "llava-hf/fake") == str(model.resolve())


def test_local_pretrained_kwargs_for_directory(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x")
    assert local_pretrained_kwargs(str(model)) == {"local_files_only": True}
    assert local_pretrained_kwargs("llava-hf/foo") == {}


def test_download_local_models_dry_run_uses_project_local_layout(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "scripts/download_local_models.sh",
            "--dry-run",
            "--model-root",
            str(tmp_path / "models"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    out = result.stdout

    assert "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" in out
    assert "llava-hf/llava-onevision-qwen2-7b-ov-hf" in out
    assert str(tmp_path / "models" / "llava-0.5b-ov") in out
    assert str(tmp_path / "models" / "llava-7b-ov") in out
    assert "export DYME_MODEL_ROOT=" in out
