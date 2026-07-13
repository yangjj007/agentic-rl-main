from __future__ import annotations

import torch

import opsd_utils.prompt_builder as prompt_builder


class _FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return messages[0]["content"][1]["text"]

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.ones((len(text), 3), dtype=torch.long)}


def test_build_teacher_prompt_batch_maps_completion_indices_to_raw_samples(monkeypatch) -> None:
    seen_prompts: list[str] = []

    def fake_context(sample, provider_names, **kwargs):
        return f"suffix:{sample['answer']}", []

    def fake_save(*args, **kwargs):
        return None

    def fake_batch(processor, payloads):
        seen_prompts.extend(payload["teacher_text"] for payload in payloads)
        return {
            "input_ids": torch.ones((len(payloads), 4), dtype=torch.long),
            "attention_mask": torch.ones((len(payloads), 4), dtype=torch.long),
            "batch_num_images": [0 for _ in payloads],
        }

    monkeypatch.setattr(prompt_builder, "build_privileged_context", fake_context)
    monkeypatch.setattr(prompt_builder, "maybe_save_privileged_images", fake_save)
    monkeypatch.setattr(prompt_builder, "_build_teacher_batch_with_oom_retry", fake_batch)

    samples = [
        {"prompt": "prompt A", "answer": "2009", "image": ""},
        {"prompt": "prompt B", "answer": "No", "image": ""},
    ]

    prompt_builder.build_teacher_prompt_batch(
        _FakeProcessor(),
        samples,
        indices=[8],
        provider_names=["format_only", "visual_facts_deplot"],
        device=torch.device("cpu"),
        expanded_count=16,
        num_generations=8,
    )

    assert seen_prompts == ["prompt B\n\nsuffix:No"]
