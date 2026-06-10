import torch

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT
from opsd_utils import debug_log as opsd_debug


def route_prompt_modes(
    acc_rewards: torch.Tensor,
    num_generations: int,
    opsd_config: dict,
    recoverable_flags: list[bool],
) -> list[int]:
    """
    Route each prompt (not each completion) to GRPO / OPSD / SFT.

    Args:
        acc_rewards: (num_prompts, num_generations)
        recoverable_flags: length num_prompts
    Returns:
        list[int] of length num_prompts with MODE_* values
    """
    threshold = opsd_config.get("gate", {}).get("correct_threshold", 0.5)
    mode_name = opsd_config.get("mode", "dyme")
    enabled = opsd_config.get("enabled", False)

    num_prompts = acc_rewards.shape[0]
    modes: list[int] = []
    opsd_debug.log(
        "mode_router",
        "route_prompt_modes enter",
        num_prompts=num_prompts,
        num_generations=num_generations,
        mode_name=mode_name,
        enabled=enabled,
        threshold=threshold,
        acc_rewards_shape=tuple(acc_rewards.shape),
        recoverable_flags=recoverable_flags,
    )

    for p in range(num_prompts):
        any_correct = (acc_rewards[p] > threshold).any().item()
        recoverable = recoverable_flags[p] if p < len(recoverable_flags) else False

        if not enabled or mode_name == "dyme":
            selected = MODE_GRPO if any_correct else MODE_SFT
        elif mode_name == "opsd_only":
            selected = MODE_OPSD
        elif mode_name == "replace_sft":
            selected = MODE_GRPO if any_correct else MODE_OPSD
        elif mode_name == "opsd_on_wrong":
            if any_correct:
                selected = MODE_GRPO
            elif recoverable:
                selected = MODE_OPSD
            else:
                selected = MODE_SFT
        elif mode_name == "grpo_opsd_joint":
            selected = MODE_GRPO if any_correct else (MODE_OPSD if recoverable else MODE_SFT)
        else:
            # trimode: OPSD replaces GRPO; wrong prompts use DyME SFT cold-start
            selected = MODE_OPSD if any_correct else MODE_SFT

        modes.append(selected)
        opsd_debug.log(
            "mode_router",
            "prompt routed",
            prompt_index=p,
            any_correct=any_correct,
            recoverable=recoverable,
            selected_mode=opsd_debug.MODE_NAMES.get(selected, selected),
            acc_rewards_row=acc_rewards[p].tolist(),
        )

    opsd_debug.log_mode_summary("mode_router", modes)
    return modes


def expand_modes_to_completions(prompt_modes: list[int], num_generations: int, batch_size: int) -> list[int]:
    """Map per-prompt mode to per-completion mode."""
    completion_modes = []
    for i in range(batch_size):
        batch_id = i // num_generations
        completion_modes.append(prompt_modes[batch_id])
    opsd_debug.log(
        "mode_router",
        "expand_modes_to_completions",
        batch_size=batch_size,
        num_generations=num_generations,
        completion_modes=[opsd_debug.MODE_NAMES.get(m, m) for m in completion_modes],
    )
    return completion_modes


def route_completion_modes(
    acc_rewards: torch.Tensor,
    num_generations: int,
    batch_size: int,
    opsd_config: dict,
    recoverable_flags: list[bool],
    format_rewards: torch.Tensor | None = None,
) -> list[int]:
    """Route each completion individually (TriMode) or expand per-prompt modes."""
    gate = opsd_config.get("gate", {})
    mode_name = opsd_config.get("mode", "dyme")
    per_completion = gate.get("per_completion_opsd", False)
    threshold = gate.get("correct_threshold", 0.5)
    require_format = gate.get("require_format_for_opsd", False)

    if mode_name == "trimode" and per_completion:
        completion_modes: list[int] = []
        num_prompts = acc_rewards.shape[0]
        for i in range(batch_size):
            prompt_idx = i // num_generations
            gen_idx = i % num_generations
            acc_ok = acc_rewards[prompt_idx, gen_idx].item() > threshold
            fmt_ok = True
            if require_format and format_rewards is not None:
                fmt_ok = format_rewards[prompt_idx, gen_idx].item() > 0
            selected = MODE_OPSD if (acc_ok and fmt_ok) else MODE_SFT
            completion_modes.append(selected)
        opsd_debug.log(
            "mode_router",
            "route_completion_modes trimode per-completion",
            batch_size=batch_size,
            num_generations=num_generations,
            per_completion_opsd=True,
            require_format_for_opsd=require_format,
            completion_modes=[opsd_debug.MODE_NAMES.get(m, m) for m in completion_modes],
        )
        return completion_modes

    prompt_modes = route_prompt_modes(
        acc_rewards, num_generations, opsd_config, recoverable_flags
    )
    return expand_modes_to_completions(prompt_modes, num_generations, batch_size)
