from __future__ import annotations

"""Unified D2 evaluation metrics and the deliberately narrow model conclusion."""

from typing import Mapping, Sequence

import numpy as np

from .metrics import distribution_metrics, hierarchical_bootstrap, summarize


def pseudobulk_pearson(real: np.ndarray, predicted: np.ndarray) -> float:
    real, predicted = map(lambda x: np.asarray(x, dtype=float), (real, predicted))
    if real.shape != predicted.shape or real.ndim != 2:
        raise ValueError("real and predicted must be equal-shaped cell × gene matrices")
    a, b = real.mean(axis=0), predicted.mean(axis=0)
    if np.std(a) <= 1e-12 and np.std(b) <= 1e-12:
        return 1.0 if float(np.mean(a) * np.mean(b)) >= 0 else 0.0
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def perturbation_discrimination(real: np.ndarray, predicted: np.ndarray,
                                perturbations: Sequence[str], *,
                                max_pairs: int | None = 10000) -> float:
    """Fraction of perturbation pairs whose predicted separation has correct order."""
    real, predicted = map(lambda x: np.asarray(x, dtype=float), (real, predicted))
    labels = np.asarray(perturbations, dtype=str)
    if real.shape != predicted.shape or len(labels) != len(real):
        raise ValueError("prediction arrays and perturbation labels do not align")
    groups = sorted(set(labels))
    if len(groups) < 2:
        return 1.0
    true_means = {group: real[labels == group].mean(axis=0) for group in groups}
    pred_means = {group: predicted[labels == group].mean(axis=0) for group in groups}
    pairs = [(i, j) for i, _ in enumerate(groups) for j in range(i + 1, len(groups))]
    if max_pairs is not None and len(pairs) > int(max_pairs):
        rng = np.random.default_rng(20260901)
        selected = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(index)] for index in selected]
    outcomes = []
    for left_index, right_index in pairs:
        left, right = groups[left_index], groups[right_index]
        true_gap = np.linalg.norm(true_means[left] - true_means[right])
        pred_gap = np.linalg.norm(pred_means[left] - pred_means[right])
        outcomes.append(float((true_gap <= 1e-12 and pred_gap <= 1e-12) or
                              (true_gap > 1e-12 and pred_gap > 0.0)))
    return float(np.mean(outcomes)) if outcomes else 1.0


def evaluate_d2_predictions(real: np.ndarray, predicted: np.ndarray,
                            perturbations: Sequence[str], conditions: Sequence[str],
                            baseline: np.ndarray | None = None,
                            ntc_mask: Sequence[bool] | None = None) -> dict:
    real, predicted = map(lambda x: np.asarray(x, dtype=float), (real, predicted))
    labels = np.asarray(perturbations, dtype=str)
    cond = np.asarray(conditions, dtype=str)
    if real.shape != predicted.shape or len(labels) != len(real) or len(cond) != len(real):
        raise ValueError("D2 evaluation arrays do not align")
    distribution_real, distribution_predicted = real, predicted
    distribution_sample_n = min(len(real), 10000)
    if len(real) > distribution_sample_n:
        sample = np.random.default_rng(20260901).choice(len(real), size=distribution_sample_n,
                                                         replace=False)
        distribution_real = real[sample]
        distribution_predicted = predicted[sample]
    result = {"pseudobulk_pearson": pseudobulk_pearson(real, predicted),
              "perturbation_discrimination": perturbation_discrimination(real, predicted, labels),
              "summary": summarize(real, predicted).__dict__,
              "distribution": distribution_metrics(distribution_real, distribution_predicted),
              "distribution_sample_n": int(distribution_sample_n),
              "by_condition": {}}
    for condition in sorted(set(cond)):
        mask = cond == condition
        result["by_condition"][condition] = {
            "n": int(mask.sum()),
            "pseudobulk_pearson": pseudobulk_pearson(real[mask], predicted[mask]),
            "summary": summarize(real[mask], predicted[mask]).__dict__,
        }
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        if baseline.shape != real.shape:
            raise ValueError("baseline shape differs from D2 response")
        result["baseline"] = {"pseudobulk_pearson": pseudobulk_pearson(real, baseline),
                               "summary": summarize(real, baseline).__dict__,
                               "distribution": distribution_metrics(real, baseline)}
        result["relative_mae_improvement"] = float(1.0 - result["summary"]["mae"] /
                                                     max(result["baseline"]["summary"]["mae"], 1e-12))
    if ntc_mask is not None:
        mask = np.asarray(ntc_mask, dtype=bool)
        if len(mask) != len(real):
            raise ValueError("NTC mask does not align")
        result["ntc_zero_effect_error"] = float(np.mean(np.abs(predicted[mask] - real[mask]))) if mask.any() else None
        result["ntc_n"] = int(mask.sum())
    return result


def conclude_model(transfer: Mapping, scratch: Mapping, baseline: Mapping,
                   transfer_global_pass: bool, scratch_global_pass: bool,
                   program_direction_evaluable: bool = True) -> dict:
    """Return only one of the four conclusions allowed by the D2 protocol."""
    if not program_direction_evaluable:
        conclusion = "一般表达预测通过，但谱系程序方向不可评价"
    elif transfer_global_pass and scratch_global_pass:
        conclusion = "Transfer" if transfer.get("primary_64_score", -np.inf) >= scratch.get("primary_64_score", -np.inf) else "Scratch"
    elif transfer_global_pass:
        conclusion = "Transfer"
    elif scratch_global_pass:
        conclusion = "Scratch"
    else:
        conclusion = "STATE未超过简单基线"
    return {"model_conclusion": conclusion,
            "transfer_global_pass": bool(transfer_global_pass),
            "scratch_global_pass": bool(scratch_global_pass),
            "baseline_name": baseline.get("name", "best_simple_baseline"),
            "MODEL_STATE_VALID": "NOT_EVALUABLE",
            "composition_and_sequential_planning": "NOT_STARTED"}
