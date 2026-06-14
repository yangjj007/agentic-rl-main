import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.teacher_batching import resolve_teacher_device_map


def test_auto_complement_two_gpus(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config.yaml")
    monkeypatch.delenv("DYME_TEACHER_DEVICE_MAP", raising=False)
    assert resolve_teacher_device_map("auto", local_rank=0, num_gpus=2) == "cuda:1"
    assert resolve_teacher_device_map("auto", local_rank=1, num_gpus=2) == "cuda:0"
    assert resolve_teacher_device_map(None, local_rank=1, num_gpus=2) == "cuda:0"


def test_fixed_cuda1_avoids_collision_on_rank1(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config.yaml")
    assert resolve_teacher_device_map("cuda:1", local_rank=1, num_gpus=2) == "cuda:0"


def test_fixed_cuda1_kept_on_rank0(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config.yaml")
    assert resolve_teacher_device_map("cuda:1", local_rank=0, num_gpus=2) == "cuda:1"


def test_same_colocate_placement():
    assert resolve_teacher_device_map("same", local_rank=1, num_gpus=2) == "cuda:1"
    assert resolve_teacher_device_map("colocate", local_rank=0, num_gpus=2) == "cuda:0"


def test_auto_colocate_under_deepspeed_config(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config_zero2.yaml")
    monkeypatch.delenv("DYME_TEACHER_DEVICE_MAP", raising=False)
    assert resolve_teacher_device_map("auto", local_rank=0, num_gpus=2) == "cuda:0"
    assert resolve_teacher_device_map("auto", local_rank=1, num_gpus=2) == "cuda:1"

