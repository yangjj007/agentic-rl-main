import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
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


def test_zero3_teacher_load_avoids_device_map_kw(monkeypatch):
    class _VisionTower:
        def __init__(self):
            self.grad_flag = None

        def requires_grad_(self, flag):
            self.grad_flag = flag
            return self

    class _BaseModel:
        def __init__(self):
            self.vision_tower = _VisionTower()

    class _Teacher:
        def __init__(self):
            self.base_model = _BaseModel()
            self.to_device = None
            self.eval_called = False
            self.grad_flag = None

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def requires_grad_(self, flag):
            self.grad_flag = flag
            return self

    captured = {}

    def _fake_from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Teacher()

    monkeypatch.setenv("ACCELERATE_CONFIG", "default_config_zero3_colocate.yaml")
    monkeypatch.setattr(main, "resolve_model_path", lambda path: path)
    monkeypatch.setattr(main, "validate_local_model_dir", lambda path, role: path)
    monkeypatch.setattr(main, "local_pretrained_kwargs", lambda path: {})
    monkeypatch.setattr(main.LlavaOnevisionForConditionalGeneration, "from_pretrained", _fake_from_pretrained)

    teacher = main.load_teacher_model(
        {
            "teacher_model_path": "/tmp/fake-teacher",
            "teacher_dtype": "bfloat16",
            "teacher_device_map": "auto",
            "use_flash_attention_2": False,
        },
        local_rank=0,
        num_gpus=2,
    )

    assert captured["args"][0] == "/tmp/fake-teacher"
    assert "device_map" not in captured["kwargs"]
    assert teacher.to_device == "cuda:0"
    assert teacher.eval_called is True
    assert teacher.grad_flag is False
    assert teacher.base_model.vision_tower.grad_flag is False
