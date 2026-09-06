from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvaluationStatus(str, Enum):
    NOT_EVALUATED = "NOT_EVALUATED"
    DATA_VALID = "DATA_VALID"
    MODEL_GLOBAL_VALID = "MODEL_GLOBAL_VALID"
    MODEL_GLOBAL_VALID_PROVISIONAL = "MODEL_GLOBAL_VALID_PROVISIONAL"
    MODEL_STATE_VALID = "MODEL_STATE_VALID"
    MODEL_STATE_VALID_PROVISIONAL = "MODEL_STATE_VALID_PROVISIONAL"
    MODEL_VALID = "MODEL_VALID"
    MODEL_VALID_PROVISIONAL = "MODEL_VALID_PROVISIONAL"
    COMPOSITION_VALID = "COMPOSITION_VALID"
    COMPOSITION_VALID_PROVISIONAL = "COMPOSITION_VALID_PROVISIONAL"
    PLANNER_VALID = "PLANNER_VALID"
    EXPERIMENT_VALID = "EXPERIMENT_VALID"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class TargetContract:
    """Versioned target definition; draft contracts cannot drive locked tests."""

    name: str = "th2_to_th17_v0"
    version: str = "draft"
    target_programs: tuple[str, ...] = ("Th17",)
    source_programs: tuple[str, ...] = ("Th2",)
    safety_programs: tuple[str, ...] = ("stress", "apoptosis", "general_activation")
    shared_gene_count: int = 2000
    reference_dataset: str = "D1"
    frozen: bool = False
    weights: dict[str, float] = field(default_factory=lambda: {
        "Th17": 1.0, "Th2": -1.0, "stress": -0.5,
        "apoptosis": -0.75, "general_activation": -0.5,
    })
    hard_constraints: dict[str, Any] = field(default_factory=lambda: {
        "Th17_min_shift": 0.0, "Th2_max_shift": 0.0,
        "safety_ntc_p95": "D1_NTC_p95",
    })
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_multiobjective(self, program_scores: dict[str, float]) -> float:
        """Score a candidate while retaining every program component."""
        scores = dict(program_scores)
        if "general_activation" not in scores and "generic_activation" in scores:
            scores["general_activation"] = scores["generic_activation"]
        missing = [name for name in self.weights if name not in scores]
        if missing:
            raise ValueError(f"missing program scores: {missing}")
        return float(sum(self.weights[name] * float(scores[name]) for name in self.weights))

    def meets_constraints(self, program_shifts: dict[str, float], ntc_p95: dict[str, float]) -> bool:
        """Apply hard safety constraints independently of the ranking score."""
        if float(program_shifts.get("Th17", float("-inf"))) <= float(self.hard_constraints["Th17_min_shift"]):
            return False
        if float(program_shifts.get("Th2", float("inf"))) > float(self.hard_constraints["Th2_max_shift"]):
            return False
        for name in ("stress", "apoptosis", "general_activation"):
            observed = program_shifts.get(name, program_shifts.get("generic_activation", float("inf")) if name == "general_activation" else float("inf"))
            limit = ntc_p95.get(name, ntc_p95.get("generic_activation", float("inf")) if name == "general_activation" else float("inf"))
            if float(observed) > float(limit):
                return False
        return True

    def freeze(self, version: str) -> "TargetContract":
        if self.frozen:
            raise ValueError("target contract is already frozen")
        if not version or version == "draft":
            raise ValueError("a non-draft version is required for freezing")
        return TargetContract(
            name=self.name,
            version=version,
            target_programs=self.target_programs,
            source_programs=self.source_programs,
            safety_programs=self.safety_programs,
            shared_gene_count=self.shared_gene_count,
            reference_dataset=self.reference_dataset,
            frozen=True,
            weights=dict(self.weights),
            hard_constraints=dict(self.hard_constraints),
            metadata=dict(self.metadata),
        )


@dataclass
class PredictionBundle:
    model: str
    gene_order: list[str]
    predicted: Any
    uncertainty: Any | None = None
    ood_score: Any | None = None
    intermediate_support: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
