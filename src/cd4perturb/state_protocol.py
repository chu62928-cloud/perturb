from __future__ import annotations

"""Split-safe orchestration rules for STATE zero-shot and light adaptation.

The module is deliberately framework-free.  An isolated STATE adapter supplies
validation metrics; this module decides whether adaptation is justified and
selects among output-layer, last-layer and LoRA candidates without ever
looking at the held-out test fold.
"""

from typing import Iterable, Mapping


ADAPTATION_METHODS = ("output_layer", "last_layers", "LoRA")


def decide_state_adaptation(zero_shot: Mapping[str, float],
                            best_simple_baseline: Mapping[str, float]) -> dict:
    """Apply the predeclared zero-shot decision rule on validation metrics."""
    required = ("standardized_rmse", "direction_accuracy")
    if any(key not in zero_shot or key not in best_simple_baseline for key in required):
        raise ValueError("STATE and baseline validation metrics must include RMSE and direction accuracy")
    rmse_worse = float(zero_shot["standardized_rmse"]) > float(best_simple_baseline["standardized_rmse"])
    direction_worse = float(zero_shot["direction_accuracy"]) < float(best_simple_baseline["direction_accuracy"])
    if rmse_worse and direction_worse:
        return {"decision": "STOP_NO_ADAPTATION_JUSTIFICATION",
                "reason": "zero-shot is inferior on both primary validation metrics",
                "selection_scope": "D2_train_validation_primary_64", "methods": []}
    return {"decision": "COMPARE_LIGHT_ADAPTATION",
            "reason": "zero-shot is not inferior on both primary validation metrics",
            "selection_scope": "D2_train_validation_primary_64",
            "methods": list(ADAPTATION_METHODS)}


def select_adaptation(candidates: Iterable[Mapping], *, evaluation_scope: str = "primary_64") -> dict:
    """Choose one adaptation using validation metrics only.

    Candidate records containing a test metric are rejected to prevent test
    leakage during method or threshold selection.
    """
    rows = [dict(row) for row in candidates]
    if not rows:
        raise ValueError("at least one adaptation candidate is required")
    if evaluation_scope != "primary_64":
        raise ValueError("STATE adaptation selection is restricted to primary_64")
    for row in rows:
        if row.get("method") not in ADAPTATION_METHODS:
            raise ValueError(f"unsupported STATE adaptation method: {row.get('method')}")
        if any(key in row for key in ("test_standardized_rmse", "test_direction_accuracy", "test_metrics")):
            raise ValueError("test metrics cannot be used during STATE adaptation selection")
        if "validation_standardized_rmse" not in row or "validation_direction_accuracy" not in row:
            raise ValueError("adaptation candidates need validation metrics")
    chosen = min(rows, key=lambda row: (float(row["validation_standardized_rmse"]),
                                         -float(row["validation_direction_accuracy"]),
                                         str(row["method"])))
    return {"selected_method": str(chosen["method"]),
            "selection_scope": "D2_train_validation_primary_64",
            "candidate_count": len(rows), "validation_metrics": {
                "standardized_rmse": float(chosen["validation_standardized_rmse"]),
                "direction_accuracy": float(chosen["validation_direction_accuracy"])}}

