from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable


@dataclass(frozen=True)
class CandidateScore:
    first: str
    second: str
    target_improvement: float
    source_decay: float
    uncertainty: float
    model_disagreement: float
    ood_distance: float
    intermediate_support: float
    stress: float
    apoptosis: float
    general_activation: float
    donor_stability: float
    total_score: float
    eligible: bool = True
    rejection_reason: str = ""


def score_candidate(parts: dict[str, float], weights: dict[str, float] | None = None) -> CandidateScore:
    weights = weights or {"target_improvement": 1.0, "source_decay": 0.5, "uncertainty": -0.5,
                          "model_disagreement": -0.5, "ood_distance": -0.5, "intermediate_support": 0.5,
                          "stress": -0.5, "apoptosis": -0.5, "general_activation": -0.25,
                          "donor_stability": 0.5}
    total = sum(weights.get(k, 0.0) * float(parts.get(k, 0.0)) for k in weights)
    return CandidateScore(total_score=total, **{k: float(parts.get(k, 0.0)) for k in CandidateScore.__dataclass_fields__ if k not in {"first", "second", "total_score", "eligible", "rejection_reason"}},
                          first=str(parts["first"]), second=str(parts["second"]))


def beam_search(actions: Iterable[str], transition: Callable[[str, str], dict[str, float]], width: int = 32) -> list[CandidateScore]:
    candidates = []
    actions = list(dict.fromkeys(actions))
    for first in actions:
        for second in actions:
            if first == second:
                continue
            parts = transition(first, second)
            candidates.append(score_candidate({**parts, "first": first, "second": second}))
    candidates.sort(key=lambda x: x.total_score, reverse=True)
    return candidates[:width]

