"""Fresh rollout replay for auxiliary GRPO-style policy-gradient updates."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RolloutReplayConfig:
    enabled: bool = False
    weight: float = 0.05
    capacity: int = 256
    batch_size: int = 2
    after_step: int = 50
    until_step: int = 0
    max_age_steps: int = 64
    min_abs_advantage: float = 0.05
    correct_threshold: float = 0.5
    priority_alpha: float = 1.0
    seed: int = 17
    positive_only: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "RolloutReplayConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            weight=max(0.0, float(raw.get("weight", 0.05) or 0.0)),
            capacity=max(0, int(raw.get("capacity", 256) or 0)),
            batch_size=max(0, int(raw.get("batch_size", 2) or 0)),
            after_step=max(0, int(raw.get("after_step", 50) or 0)),
            until_step=max(0, int(raw.get("until_step", 0) or 0)),
            max_age_steps=max(0, int(raw.get("max_age_steps", 64) or 0)),
            min_abs_advantage=max(0.0, float(raw.get("min_abs_advantage", 0.05) or 0.0)),
            correct_threshold=float(raw.get("correct_threshold", 0.5) or 0.5),
            priority_alpha=max(0.0, float(raw.get("priority_alpha", 1.0) or 0.0)),
            seed=int(raw.get("seed", 17) or 17),
            positive_only=bool(raw.get("positive_only", True)),
        )


@dataclass
class RolloutReplayEntry:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_per_token_logps: torch.Tensor
    advantage: float
    acc_reward: float
    global_step: int
    pixel_values: torch.Tensor | None = None
    image_sizes: torch.Tensor | None = None


@dataclass
class RolloutReplayAddStats:
    seen: int = 0
    added: int = 0
    skipped_not_positive: int = 0
    skipped_low_advantage: int = 0
    skipped_empty: int = 0


def stack_optional_compatible_tensors(
    rows: list[torch.Tensor | None],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    kept = [row for row in rows if isinstance(row, torch.Tensor)]
    if not kept or len(kept) != len(rows):
        return None
    shape = tuple(kept[0].shape)
    if any(tuple(row.shape) != shape for row in kept):
        return None
    if device is None:
        return torch.stack(kept, dim=0)
    return torch.stack([row.to(device) for row in kept], dim=0)


def _clone_row(value: torch.Tensor | None, row: int) -> torch.Tensor | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 0:
        return value.detach().cpu().clone()
    if row >= int(value.shape[0]):
        return None
    return value[row].detach().cpu().clone()


class RolloutReplayBuffer:
    def __init__(self, config: RolloutReplayConfig, *, process_index: int = 0):
        self.config = config
        self.entries: deque[RolloutReplayEntry] = deque(maxlen=max(0, int(config.capacity)))
        self._rng = random.Random(config.seed + 7919 * int(process_index))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def available(self) -> bool:
        cfg = self.config
        return bool(cfg.enabled and cfg.weight > 0 and cfg.capacity > 0 and cfg.batch_size > 0)

    def enabled_for_step(self, global_step: int) -> bool:
        if not self.available:
            return False
        step = int(global_step)
        if step < self.config.after_step:
            return False
        if self.config.until_step > 0 and step > self.config.until_step:
            return False
        return True

    def _is_fresh(self, entry: RolloutReplayEntry, global_step: int) -> bool:
        if self.config.max_age_steps <= 0:
            return True
        return int(global_step) - int(entry.global_step) <= self.config.max_age_steps

    def prune(self, global_step: int) -> int:
        before = len(self.entries)
        if self.config.max_age_steps > 0:
            self.entries = deque(
                [entry for entry in self.entries if self._is_fresh(entry, global_step)],
                maxlen=self.entries.maxlen,
            )
        return before - len(self.entries)

    def add_entry(self, entry: RolloutReplayEntry) -> None:
        if not self.available:
            return
        self.entries.append(entry)

    def add_batch(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        advantages: torch.Tensor,
        acc_rewards: torch.Tensor | None,
        global_step: int,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
    ) -> RolloutReplayAddStats:
        stats = RolloutReplayAddStats()
        if not self.available:
            return stats
        batch = int(completion_ids.shape[0])
        flat_adv = advantages.detach().float().view(batch, -1)[:, 0].cpu()
        flat_acc = (
            acc_rewards.detach().float().view(-1).cpu()
            if isinstance(acc_rewards, torch.Tensor)
            else torch.ones(batch, dtype=torch.float32)
        )
        for row in range(batch):
            stats.seen += 1
            mask_sum = float(completion_mask[row].detach().float().sum().item())
            if mask_sum <= 0:
                stats.skipped_empty += 1
                continue
            adv = float(flat_adv[row].item())
            acc = float(flat_acc[row].item()) if row < int(flat_acc.numel()) else 0.0
            if self.config.positive_only and (adv <= 0 or acc <= self.config.correct_threshold):
                stats.skipped_not_positive += 1
                continue
            if abs(adv) < self.config.min_abs_advantage:
                stats.skipped_low_advantage += 1
                continue
            self.entries.append(
                RolloutReplayEntry(
                    prompt_ids=prompt_ids[row].detach().cpu().clone(),
                    prompt_mask=prompt_mask[row].detach().cpu().clone(),
                    completion_ids=completion_ids[row].detach().cpu().clone(),
                    completion_mask=completion_mask[row].detach().cpu().clone(),
                    old_per_token_logps=old_per_token_logps[row].detach().cpu().clone(),
                    advantage=adv,
                    acc_reward=acc,
                    global_step=int(global_step),
                    pixel_values=_clone_row(pixel_values, row),
                    image_sizes=_clone_row(image_sizes, row),
                )
            )
            stats.added += 1
        return stats

    def sample(self, *, global_step: int) -> list[RolloutReplayEntry]:
        if not self.enabled_for_step(global_step):
            return []
        self.prune(global_step)
        candidates = list(self.entries)
        if not candidates:
            return []
        count = min(self.config.batch_size, len(candidates))
        if self.config.priority_alpha <= 0:
            return self._rng.sample(candidates, count)
        weights = [
            max(abs(float(entry.advantage)), 1e-6) ** self.config.priority_alpha
            for entry in candidates
        ]
        selected: list[RolloutReplayEntry] = []
        pool = list(candidates)
        pool_weights = list(weights)
        for _ in range(count):
            total = sum(pool_weights)
            if total <= 0:
                idx = self._rng.randrange(len(pool))
            else:
                marker = self._rng.random() * total
                acc = 0.0
                idx = 0
                for i, weight in enumerate(pool_weights):
                    acc += weight
                    if acc >= marker:
                        idx = i
                        break
            selected.append(pool.pop(idx))
            pool_weights.pop(idx)
            if not pool:
                break
        return selected
