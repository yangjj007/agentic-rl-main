from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch

from opsd_utils.constants import MODE_OPSD, MODE_SKIP
from opsd_utils.teacher_probe_agreement import decide_teacher_probe_agreement
from trainer.DyMETrainer import DyMETrainer

trainer_module = importlib.import_module("trainer.DyMETrainer")


def test_teacher_probe_agreement_accepts_only_matching_normalized_answers() -> None:
    decision = decide_teacher_probe_agreement(
        outputs=[
            "Answer: Latvia and Australia",
            "Answer: Latvia, Australia",
            "Answer: [Latvia, Australia]",
        ],
        reference="[Latvia, Australia]",
        answer_flag="answer:",
        max_relative_change=0.05,
    )

    assert decision.agreement_accepted is True
    assert decision.verified_correct is True
    assert decision.reason_code == "agreement_verified_correct"
    assert decision.selected_output == "Answer: Latvia and Australia"
    assert decision.normalized_answer == "latvia,australia"


def test_teacher_probe_agreement_rejects_disagreement_before_reference_verification() -> None:
    decision = decide_teacher_probe_agreement(
        outputs=["Answer: 70", "Answer: 70", "Answer: 72"],
        reference="70",
        answer_flag="answer:",
        max_relative_change=0.05,
    )

    assert decision.agreement_accepted is False
    assert decision.verified_correct is False
    assert decision.reason_code == "answer_disagreement"
    assert decision.selected_output == "Answer: 70"


class _Tokenizer:
    eos_token_id = 2


class _Processor:
    tokenizer = _Tokenizer()


def _trainer_for_agreement(monkeypatch, outputs_by_profile: dict[str, list[str]]) -> DyMETrainer:
    trainer = DyMETrainer.__new__(DyMETrainer)
    trainer.opsd_config = {
        "mode": "dyme_teacher_probe_opd",
        "teacher_probe": {
            "enabled": True,
            "context_providers": ["format_only"],
            "prompt_profile": "chartqa_visual_answer_prefix",
            "skip_no_evidence": False,
            "agreement_gate": {
                "enabled": True,
                "prompt_profiles": [
                    "chartqa_visual_answer_prefix",
                    "chartqa_visual_short",
                    "chartqa_visual_answer_prefix_numeric",
                ],
            },
        },
        "gate": {"teacher_probe_failure_route": "mixed_grpo_all_wrong_skip"},
    }
    trainer.teacher_model = object()
    trainer.processing_class = _Processor()
    trainer.num_generations = 1
    trainer.args = SimpleNamespace(output_dir=None)
    trainer.accelerator = SimpleNamespace(process_index=0, is_main_process=False)
    trainer._teacher_probe_preview_logged = False
    trainer._perf_start = lambda: 0.0
    trainer._perf_elapsed = lambda _start: 0.0

    built_profiles: list[str] = []

    def fake_build_teacher_prompt_batch(
        _processor,
        _inputs,
        indices,
        provider_names,
        device,
        *,
        opsd_config,
        **_kwargs,
    ):
        profile = opsd_config["teacher_probe"]["prompt_profile"]
        built_profiles.append(profile)
        return {
            "profile": profile,
            "teacher_prompt_ids": torch.ones(len(indices), 2, dtype=torch.long, device=device),
            "teacher_prompt_mask": torch.ones(len(indices), 2, dtype=torch.long, device=device),
            "teacher_stats": {},
        }

    def fake_generate(self, teacher_tensors, rows, **_kwargs):
        profile = teacher_tensors["profile"]
        outputs = outputs_by_profile[profile]
        return [
            (
                torch.tensor([10 + row], dtype=torch.long),
                torch.ones(1, dtype=torch.long),
                outputs[row],
            )
            for row in rows
        ], False

    monkeypatch.setattr(
        trainer_module,
        "build_teacher_prompt_batch",
        fake_build_teacher_prompt_batch,
    )
    monkeypatch.setattr(DyMETrainer, "_teacher_generate_batch_from_tensors", fake_generate)
    trainer._built_profiles = built_profiles
    return trainer


def test_teacher_probe_routing_accepts_agreed_verified_teacher(monkeypatch) -> None:
    trainer = _trainer_for_agreement(
        monkeypatch,
        {
            "chartqa_visual_answer_prefix": ["Answer: 70"],
            "chartqa_visual_short": ["Answer: 70"],
            "chartqa_visual_answer_prefix_numeric": ["Answer: 70"],
        },
    )

    modes, teacher_trajs, texts, stats = trainer._apply_teacher_probe_routing(
        inputs=[{"prompt": "q", "answer": "70", "image": "chart.png"}],
        completion_modes=[MODE_OPSD],
        acc_rewards=torch.tensor([0.0]),
        answers=["70"],
        completions=["Answer: 0"],
        answer_flag="answer:",
        global_step=1,
        device=torch.device("cpu"),
        group_has_correct=[False],
        group_reward_std=[0.0],
    )

    assert trainer._built_profiles == [
        "chartqa_visual_answer_prefix",
        "chartqa_visual_short",
        "chartqa_visual_answer_prefix_numeric",
    ]
    assert modes == [MODE_OPSD]
    assert stats["teacher_probe_correct"] == 1
    assert stats["teacher_probe_agreement_accepted"] == 1
    assert stats["teacher_probe_agreement_rejected"] == 0
    assert teacher_trajs[0][0].tolist() == [10]
    assert texts[0] == "Answer: 70"


def test_teacher_probe_routing_rejects_disagreed_teacher(monkeypatch) -> None:
    trainer = _trainer_for_agreement(
        monkeypatch,
        {
            "chartqa_visual_answer_prefix": ["Answer: 70"],
            "chartqa_visual_short": ["Answer: 72"],
            "chartqa_visual_answer_prefix_numeric": ["Answer: 70"],
        },
    )

    modes, teacher_trajs, texts, stats = trainer._apply_teacher_probe_routing(
        inputs=[{"prompt": "q", "answer": "70", "image": "chart.png"}],
        completion_modes=[MODE_OPSD],
        acc_rewards=torch.tensor([0.0]),
        answers=["70"],
        completions=["Answer: 0"],
        answer_flag="answer:",
        global_step=1,
        device=torch.device("cpu"),
        group_has_correct=[False],
        group_reward_std=[0.0],
    )

    assert modes == [MODE_SKIP]
    assert stats["teacher_probe_correct"] == 0
    assert stats["teacher_probe_wrong"] == 1
    assert stats["teacher_probe_agreement_accepted"] == 0
    assert stats["teacher_probe_agreement_rejected"] == 1
    assert stats["teacher_probe_agreement_reject_reasons"]["answer_disagreement"] == 1
    assert teacher_trajs == {}
    assert texts == {}
