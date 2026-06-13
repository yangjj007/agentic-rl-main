"""Regression tests for completion degeneration heuristics."""
from unittest.mock import patch

import torch

from opsd_utils import debug_log as opsd_debug
from opsd_utils.diagnostics import (
    _detect_char_repeat,
    _detect_degeneration,
    _detect_repeat_loop,
    _detect_single_token_repeat,
    _max_same_token_run,
    is_degenerate_completion,
    log_generate_probe,
)


class _FakeTokenizer:
    eos_token_id = 151645
    pad_token_id = 151643
    bos_token_id = None

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(i) for i in ids)


def test_single_token_repeat_detects_cjk_loop():
    ids = [39992, 25, 7379] + [41146] * 40
    assert _detect_single_token_repeat(ids)
    assert _max_same_token_run(ids) == (40, 41146)


def test_ngram_repeat_not_limited_to_first_eight_tokens():
    prefix = list(range(20))
    gram = [9, 8, 7]
    ids = prefix + gram * 5
    assert _detect_repeat_loop(ids)


def test_char_repeat_detects_qiqiqi():
    assert _detect_char_repeat("其其其其其其")


def test_is_degenerate_completion_detects_repeat():
    ids = [39992, 25] + [41146] * 20
    assert is_degenerate_completion(ids, "Goal: x\n" + "其" * 40)


def test_short_numeric_answer_not_degenerate_without_answer_flag():
    ids = [198, 17, 15, 18, 15]  # \n2030
    assert not is_degenerate_completion(ids, "\n2030", require_answer_flag=False)
    assert is_degenerate_completion(ids, "\n2030", require_answer_flag=True)


def test_degeneration_flags_missing_answer():
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    text = "Goal: test\nObservation: x\nReasoning: y\nConclusion: z"
    is_deg, reasons = _detect_degeneration(ids, text, answer_flag="Answer:")
    assert is_deg
    assert any(r.startswith("ANSWER_FLAG_COUNT") for r in reasons)


def test_log_generate_probe_does_not_shadow_tokenizer_across_samples():
    """sample[0] single-token repeat must not break decode for sample[1]."""
    repeat_tail = [24] * 12
    row0 = [39992, 25, 7379] + repeat_tail + [0] * (200 - 3 - len(repeat_tail))
    row1 = [39992, 25, 7379, 100, 101, 102] + [0] * 194
    completion_ids = torch.tensor([row0, row1], dtype=torch.long)
    completion_mask = torch.tensor(
        [[1] * 15 + [0] * 185, [1] * 6 + [0] * 194],
        dtype=torch.long,
    )
    is_eos = torch.zeros_like(completion_mask, dtype=torch.bool)
    is_eos[:, 14] = True
    is_eos[:, 5] = True
    eos_idx = torch.tensor([14, 5], dtype=torch.long)
    completions = ["Goal: repeat\n" + "x" * 20, "Goal: ok\nAnswer: 1"]

    with patch.object(opsd_debug, "should_log_probe", return_value=True):
        stats = log_generate_probe(
            global_step=1,
            trainer_step=1,
            prompt_length=100,
            prompt_completion_ids=torch.zeros(2, 300, dtype=torch.long),
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            is_eos=is_eos,
            eos_idx=eos_idx,
            completions=completions,
            tokenizer=_FakeTokenizer(),
            generation_config=None,
            max_completion_length=200,
            num_generations=1,
            sample_count=2,
        )
    assert stats["degenerate_count"] >= 1
