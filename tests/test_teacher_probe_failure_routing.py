from __future__ import annotations

import pytest

from opsd_utils.constants import MODE_GRPO, MODE_SFT, MODE_SKIP
from opsd_utils.signal_aware_routing import teacher_probe_failure_mode


def test_teacher_probe_failure_route_preserves_legacy_sft() -> None:
    assert teacher_probe_failure_mode(group_has_correct=False, route="sft") == MODE_SFT
    assert teacher_probe_failure_mode(group_has_correct=True, route="sft") == MODE_SFT


def test_teacher_probe_failure_route_uses_grpo_for_mixed_groups() -> None:
    assert (
        teacher_probe_failure_mode(
            group_has_correct=True,
            route="mixed_grpo_all_wrong_skip",
        )
        == MODE_GRPO
    )


def test_teacher_probe_failure_route_skips_all_wrong_groups() -> None:
    assert (
        teacher_probe_failure_mode(
            group_has_correct=False,
            route="mixed_grpo_all_wrong_skip",
        )
        == MODE_SKIP
    )


def test_teacher_probe_failure_route_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="teacher probe failure route"):
        teacher_probe_failure_mode(group_has_correct=False, route="answer_only")
