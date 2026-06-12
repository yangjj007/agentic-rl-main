"""Ensure anti-leakage privileged contexts do not embed gold answers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opsd_utils.leakage import privileged_suffix_has_gold
from opsd_utils.privileged import build_privileged_context
from opsd_utils.privileged.providers import FormatOnlyProvider, TextProvider


def test_format_only_has_no_gold():
    sample = {"hint": "secret chain", "answer": "Answer: 42"}
    suffix, _ = build_privileged_context(
        sample,
        ["format_only"],
        privileged_profile="text",
        opsd_config={"text_include_gold": False},
    )
    assert "42" not in suffix
    assert "secret" not in suffix
    assert "Goal:" in suffix or "structure" in suffix.lower()
    assert not privileged_suffix_has_gold(suffix, sample)


def test_text_provider_respects_include_gold_false():
    provider = TextProvider(include_gold=False)
    sample = {"hint": "step", "answer": "Answer: 1"}
    assert provider.build_teacher_suffix(sample) == ""


def test_empty_providers_no_gold():
    sample = {"hint": "step", "answer": "Answer: 99"}
    suffix, images = build_privileged_context(
        sample,
        [],
        privileged_profile="text",
        opsd_config={"text_include_gold": False},
    )
    assert suffix.strip() == ""
    assert images == [] or len(images) <= 1
    assert not privileged_suffix_has_gold(suffix, sample)


if __name__ == "__main__":
    test_format_only_has_no_gold()
    test_text_provider_respects_include_gold_false()
    test_empty_providers_no_gold()
    print("No-gold suffix tests passed.")
