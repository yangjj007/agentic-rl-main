import torch

from opsd_utils.constants import MODE_GRPO, MODE_OPSD, MODE_SFT


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

    for p in range(num_prompts):
        any_correct = (acc_rewards[p] > threshold).any().item()
        recoverable = recoverable_flags[p] if p < len(recoverable_flags) else False

        if not enabled or mode_name == "dyme":
            modes.append(MODE_GRPO if any_correct else MODE_SFT)
            continue

        if mode_name == "opsd_only":
            modes.append(MODE_OPSD)
            continue

        if mode_name == "replace_sft":
            modes.append(MODE_GRPO if any_correct else MODE_OPSD)
            continue

        if mode_name == "opsd_on_wrong":
            if any_correct:
                modes.append(MODE_GRPO)
            elif recoverable:
                modes.append(MODE_OPSD)
            else:
                modes.append(MODE_SFT)
            continue

        if mode_name == "grpo_opsd_joint":
            modes.append(MODE_GRPO if any_correct else (MODE_OPSD if recoverable else MODE_SFT))
            continue

        # trimode (default when enabled)
        if any_correct:
            modes.append(MODE_GRPO)
        elif recoverable:
            modes.append(MODE_OPSD)
        else:
            modes.append(MODE_SFT)

    return modes


def expand_modes_to_completions(prompt_modes: list[int], num_generations: int, batch_size: int) -> list[int]:
    """Map per-prompt mode to per-completion mode."""
    completion_modes = []
    for i in range(batch_size):
        batch_id = i // num_generations
        completion_modes.append(prompt_modes[batch_id])
    return completion_modes
