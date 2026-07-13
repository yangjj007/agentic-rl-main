from __future__ import annotations

from trainer.DyMETrainer import DyMETrainer


def _trainer_with_gate(gate: dict[str, object]) -> DyMETrainer:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer.opsd_config = {"gate": gate}
    trainer._in_sft_cold_start = lambda: False
    return trainer


def test_no_full_hint_mode_disables_forced_sft_replacement() -> None:
    trainer = _trainer_with_gate(
        {
            "disable_online_sft_slots": True,
            "online_sft_on_all_wrong": False,
            "teacher_probe_failure_route": "mixed_grpo_all_wrong_skip",
        }
    )

    assert trainer._should_force_sft_replace(0, ["short malformed output"], "Answer:") is False


def test_legacy_mode_preserves_forced_sft_replacement() -> None:
    trainer = _trainer_with_gate({})

    assert trainer._should_force_sft_replace(0, ["short malformed output"], "Answer:") is True
