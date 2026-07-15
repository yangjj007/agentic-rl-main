"""Reference-free runtime contracts for evidence-recovery harnesses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceAction(str, Enum):
    VISUAL_BASE = "visual_base"
    ATTACH_DEPLOT = "attach_deplot"
    VISUAL_RECOVERY = "visual_recovery"


class HarnessStatus(str, Enum):
    ACTIVE = "active"
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FATAL_FAILURE = "fatal_failure"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class EvidenceCandidate:
    attempt_id: str
    action: EvidenceAction
    answer: str | None
    raw_output: str
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    operation: str | None = None
    unresolved_refs: tuple[str, ...] = field(default_factory=tuple)
    confidence: float | None = None
    support_relation: str = "unknown"
    parse_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ValidationResult:
    validator_id: str
    status: str
    reason_code: str
    supporting_refs: tuple[str, ...] = field(default_factory=tuple)
    conflict_refs: tuple[str, ...] = field(default_factory=tuple)
    deterministic_value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class HarnessAttempt:
    candidate: EvidenceCandidate
    validation: ValidationResult
    cost: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "validation": self.validation.to_dict(),
            "cost": self.cost,
        }


@dataclass(frozen=True)
class HarnessDecision:
    status: HarnessStatus
    reason_code: str
    remaining_budget: int
    selected_attempt_id: str | None = None
    next_action: EvidenceAction | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))
