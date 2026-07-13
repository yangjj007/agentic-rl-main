from types import SimpleNamespace

import pytest


def _raising_check():
    raise ValueError("blocked")


def test_trusted_torch_load_patch_requires_explicit_env(monkeypatch):
    from opsd_utils.trusted_torch_load import maybe_allow_trusted_torch_load

    fake_trainer = SimpleNamespace(check_torch_load_is_safe=_raising_check)
    fake_import_utils = SimpleNamespace(check_torch_load_is_safe=_raising_check)
    monkeypatch.setattr(
        "opsd_utils.trusted_torch_load.importlib.import_module",
        lambda name: {
            "transformers.trainer": fake_trainer,
            "transformers.utils.import_utils": fake_import_utils,
        }[name],
    )

    enabled = maybe_allow_trusted_torch_load(
        resume_from_checkpoint="checkpoint-147",
        env={},
        log=lambda _: None,
    )

    assert enabled is False
    with pytest.raises(ValueError, match="blocked"):
        fake_trainer.check_torch_load_is_safe()
    with pytest.raises(ValueError, match="blocked"):
        fake_import_utils.check_torch_load_is_safe()


def test_trusted_torch_load_patch_disables_transformers_check_when_enabled(monkeypatch):
    from opsd_utils.trusted_torch_load import maybe_allow_trusted_torch_load

    fake_trainer = SimpleNamespace(check_torch_load_is_safe=_raising_check)
    fake_import_utils = SimpleNamespace(check_torch_load_is_safe=_raising_check)
    monkeypatch.setattr(
        "opsd_utils.trusted_torch_load.importlib.import_module",
        lambda name: {
            "transformers.trainer": fake_trainer,
            "transformers.utils.import_utils": fake_import_utils,
        }[name],
    )
    messages = []

    enabled = maybe_allow_trusted_torch_load(
        resume_from_checkpoint="checkpoint-147",
        env={"DYME_TRUST_LOCAL_TORCH_LOAD": "1"},
        log=messages.append,
    )

    assert enabled is True
    fake_trainer.check_torch_load_is_safe()
    fake_import_utils.check_torch_load_is_safe()
    assert messages
