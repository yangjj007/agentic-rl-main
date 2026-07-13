"""Positive replay buffer helpers for auxiliary online SFT mixing."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_utils.rl_prompt import PROMPT_TEMPLATE


@dataclass(frozen=True)
class PositiveReplayConfig:
    enabled: bool = False
    dataset_path: str = ""
    batch_size: int = 1
    weight: float = 0.1
    after_step: int = 0
    until_step: int = 0
    seed: int = 13
    max_rows: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "PositiveReplayConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            dataset_path=str(raw.get("dataset_path", raw.get("path", "")) or ""),
            batch_size=max(0, int(raw.get("batch_size", 1) or 0)),
            weight=max(0.0, float(raw.get("weight", 0.1) or 0.0)),
            after_step=max(0, int(raw.get("after_step", 0) or 0)),
            until_step=max(0, int(raw.get("until_step", 0) or 0)),
            seed=int(raw.get("seed", 13) or 13),
            max_rows=max(0, int(raw.get("max_rows", 0) or 0)),
        )


def _load_replay_rows(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
        return rows
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        for key in ("data", "records", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _answer_text(row: dict[str, Any]) -> str:
    target = str(row.get("target") or "").strip()
    if target:
        return target
    hint = str(row.get("hint") or "").strip()
    answer = str(row.get("answer") or row.get("reference") or "").strip()
    if hint and answer:
        return f"{hint}\nAnswer: {answer}".strip()
    if answer:
        return f"Answer: {answer}".strip()
    return hint


def replay_row_to_sft_sample(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or row.get("question_wo_prompt") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        prompt = PROMPT_TEMPLATE.format(question=question)
    sample = dict(row)
    sample["prompt"] = prompt
    sample["question_wo_prompt"] = question
    sample["answer"] = _answer_text(row)
    return sample


class PositiveReplayBuffer:
    def __init__(self, config: PositiveReplayConfig, *, process_index: int = 0):
        self.config = config
        self._rng = random.Random(config.seed + 1009 * int(process_index))
        rows = _load_replay_rows(config.dataset_path) if config.enabled else []
        if config.max_rows > 0:
            rows = rows[: config.max_rows]
        self.rows = rows

    @property
    def available(self) -> bool:
        return bool(self.config.enabled and self.config.weight > 0 and self.config.batch_size > 0 and self.rows)

    def enabled_for_step(self, global_step: int) -> bool:
        if not self.available:
            return False
        step = int(global_step)
        if step < self.config.after_step:
            return False
        if self.config.until_step > 0 and step > self.config.until_step:
            return False
        return True

    def sample(self, *, global_step: int) -> list[dict[str, Any]]:
        if not self.enabled_for_step(global_step):
            return []
        if self.config.batch_size >= len(self.rows):
            selected = list(self.rows)
            self._rng.shuffle(selected)
            selected = selected[: self.config.batch_size]
        else:
            selected = self._rng.sample(self.rows, self.config.batch_size)
        return [replay_row_to_sft_sample(row) for row in selected]
