"""Quality-gate helpers for ChartQA teacher process supervision."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from reward_utils.chart_cot_verifier import (
    ChartCoTVerification,
    verifier_error_result,
    verify_chart_cot_trajectory,
)


@dataclass(frozen=True)
class ChartCoTQualityGateConfig:
    enabled: bool = False
    mode: str = "off"
    require_quality: str = "Q3"
    log_samples: bool = True
    max_log_samples: int = 8

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "ChartCoTQualityGateConfig":
        raw = raw or {}
        mode = str(raw.get("mode", "off") or "off").lower()
        enabled = bool(raw.get("enabled", mode != "off"))
        return cls(
            enabled=enabled,
            mode=mode if mode in {"off", "diagnostic", "gate"} else "off",
            require_quality=str(raw.get("require_quality", "Q3") or "Q3").upper(),
            log_samples=bool(raw.get("log_samples", True)),
            max_log_samples=max(0, int(raw.get("max_log_samples", 8) or 0)),
        )

    @property
    def gate_active(self) -> bool:
        return self.enabled and self.mode == "gate"


@dataclass(frozen=True)
class ChartCoTQualityGateResult:
    eligible_indices: set[int]
    rejected_indices: set[int]
    metrics: dict[str, float]


@dataclass(frozen=True)
class TeacherTrajectoryQualityEvaluation:
    verifications: dict[int, ChartCoTVerification]
    gate_result: ChartCoTQualityGateResult
    sample_records: tuple[dict[str, Any], ...]


def aggregate_chart_cot_verifications(
    verifications: Mapping[int, ChartCoTVerification],
) -> dict[str, float]:
    values = list(verifications.values())
    candidate_count = len(values)
    denom = max(candidate_count, 1)
    claims = [claim for value in values for claim in value.grounded_claims]
    claim_denom = max(len(claims), 1)
    checks = [check for value in values for check in value.reasoning_checks]
    check_denom = max(len(checks), 1)

    metrics: dict[str, float] = {
        "candidate_count": float(candidate_count),
        "structure_valid_rate": sum(value.parsed.structure_valid for value in values) / denom,
        "deplot_available_rate": sum(value.deplot_available for value in values) / denom,
        "grounded_claim_count": float(len(claims)),
        "verifier_error_rate": sum(value.verification_error for value in values) / denom,
    }
    for quality in ("Q3", "Q2", "Q1", "Q0"):
        metrics[f"{quality.lower()}_rate"] = sum(value.quality == quality for value in values) / denom
    for status in ("supported", "contradicted", "unknown"):
        metrics[f"{status}_claim_rate"] = sum(claim.status == status for claim in claims) / claim_denom
    for status in ("valid", "invalid", "unknown"):
        metrics[f"reasoning_{status}_rate"] = sum(check.status == status for check in checks) / check_denom
    for status in ("consistent", "inconsistent", "unknown"):
        metrics[f"conclusion_answer_{status}_rate"] = (
            sum(value.conclusion_answer.status == status for value in values) / denom
        )
    return metrics


def filter_quality_eligible_indices(
    candidate_indices: set[int],
    verifications: Mapping[int, ChartCoTVerification],
    config: ChartCoTQualityGateConfig,
) -> ChartCoTQualityGateResult:
    candidates = {int(index) for index in candidate_indices}
    if config.gate_active:
        eligible = {
            index
            for index in candidates
            if index in verifications and verifications[index].quality == config.require_quality
        }
    else:
        eligible = set(candidates)
    rejected = candidates - eligible
    metrics = aggregate_chart_cot_verifications(verifications)
    metrics.update(
        {
            "enabled": float(config.enabled),
            "gate_active": float(config.gate_active),
            "teacher_traj_accepted_rate": len(eligible) / max(len(candidates), 1),
            "teacher_traj_rejected_count": float(len(rejected)),
        }
    )
    return ChartCoTQualityGateResult(
        eligible_indices=eligible,
        rejected_indices=rejected,
        metrics=metrics,
    )


def evaluate_teacher_trajectory_quality(
    *,
    teacher_traj_texts: Mapping[int, str],
    samples: Sequence[Mapping[str, Any]],
    num_generations: int,
    config: ChartCoTQualityGateConfig,
) -> TeacherTrajectoryQualityEvaluation:
    candidate_indices = {int(index) for index in teacher_traj_texts}
    if not config.enabled:
        gate_result = filter_quality_eligible_indices(candidate_indices, {}, config)
        return TeacherTrajectoryQualityEvaluation({}, gate_result, ())

    verifications: dict[int, ChartCoTVerification] = {}
    for global_idx in sorted(candidate_indices):
        prompt_idx = global_idx // max(int(num_generations), 1)
        sample = samples[prompt_idx] if prompt_idx < len(samples) else {}
        response = teacher_traj_texts.get(global_idx, "")
        try:
            verifications[global_idx] = verify_chart_cot_trajectory(
                response,
                sample.get("visual_fact_deplot"),
                answer_correct=True,
            )
        except Exception:
            verifications[global_idx] = verifier_error_result(
                response,
                answer_correct=True,
            )

    gate_result = filter_quality_eligible_indices(candidate_indices, verifications, config)
    records: list[dict[str, Any]] = []
    if config.log_samples and config.max_log_samples > 0:
        for global_idx in sorted(candidate_indices)[: config.max_log_samples]:
            prompt_idx = global_idx // max(int(num_generations), 1)
            sample = samples[prompt_idx] if prompt_idx < len(samples) else {}
            verification = verifications[global_idx]
            records.append(
                {
                    "global_idx": global_idx,
                    "prompt_idx": prompt_idx,
                    "question": str(sample.get("question") or sample.get("question_wo_prompt") or ""),
                    "quality": verification.quality,
                    "reason_codes": list(verification.reason_codes),
                    "structure_valid": verification.parsed.structure_valid,
                    "deplot_available": verification.deplot_available,
                    "grounded_claims": [asdict(claim) for claim in verification.grounded_claims],
                    "reasoning_checks": [asdict(check) for check in verification.reasoning_checks],
                    "conclusion_answer": asdict(verification.conclusion_answer),
                    "teacher_response": teacher_traj_texts.get(global_idx, ""),
                }
            )
    return TeacherTrajectoryQualityEvaluation(
        verifications=verifications,
        gate_result=gate_result,
        sample_records=tuple(records),
    )


def append_quality_sample_records(
    *,
    output_dir: str,
    rank: int,
    global_step: int,
    records: Sequence[Mapping[str, Any]],
) -> str | None:
    if not output_dir or not records:
        return None
    directory = os.path.join(output_dir, "chart_cot_quality")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"rank{int(rank)}.jsonl")
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            payload = {"global_step": int(global_step), **dict(record)}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path
