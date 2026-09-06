from __future__ import annotations

import numpy as np


def model_disagreement(predictions: dict[str, np.ndarray]) -> dict:
    """Return robust disagreement signals; never silently average models."""
    if not predictions:
        raise ValueError("at least one model prediction is required")
    names = sorted(predictions)
    stack = np.stack([np.asarray(predictions[n], dtype=float) for n in names], axis=0)
    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)
    directions = np.sign(stack)
    direction_conflict = np.any(directions != directions[0:1], axis=0)
    return {"models": names, "median_shape": list(median.shape),
            "mean_mad": float(np.mean(mad)), "max_mad": float(np.max(mad)),
            "direction_conflict_fraction": float(np.mean(direction_conflict))}


def robust_rank(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Rank candidates by median model rank, not arithmetic output average."""
    names = sorted(scores)
    ranks = []
    for name in names:
        values = np.asarray(scores[name], dtype=float)
        ranks.append(np.argsort(np.argsort(-values)))
    return np.median(np.stack(ranks, axis=0), axis=0)

