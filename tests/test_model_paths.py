"""Tests for local model path resolution."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.paths import (
    local_pretrained_kwargs,
    resolve_model_path,
    validate_local_model_dir,
)


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


def test_local_pretrained_kwargs_for_directory(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x")
    assert local_pretrained_kwargs(str(model)) == {"local_files_only": True}
    assert local_pretrained_kwargs("llava-hf/foo") == {}
