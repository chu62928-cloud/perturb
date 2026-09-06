from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    name: str
    prediction: np.ndarray
    metadata: dict


def ntc_no_change(x_control: np.ndarray, n: int | None = None) -> BaselineResult:
    x_control = np.asarray(x_control, dtype=float)
    return BaselineResult("ntc_no_change", np.repeat(x_control.mean(axis=0, keepdims=True), n or len(x_control), axis=0), {})


def condition_mean_effect(x_control: np.ndarray, y_perturbed: np.ndarray) -> BaselineResult:
    x_control, y_perturbed = map(lambda x: np.asarray(x, dtype=float), (x_control, y_perturbed))
    if x_control.shape[1] != y_perturbed.shape[1]:
        raise ValueError("control and perturbed feature counts differ")
    effect = y_perturbed.mean(axis=0) - x_control.mean(axis=0)
    return BaselineResult("condition_mean_effect", y_perturbed.mean(axis=0, keepdims=True), {"effect": effect})


def gene_effect_transfer(x_control: np.ndarray, gene_effect: np.ndarray,
                         n: int | None = None) -> BaselineResult:
    """Transfer a frozen gene-level effect vector from the development donor."""
    x_control = np.asarray(x_control, dtype=float)
    effect = np.asarray(gene_effect, dtype=float)
    if x_control.ndim != 2 or effect.ndim != 1 or x_control.shape[1] != effect.shape[0]:
        raise ValueError("gene-effect transfer expects control matrix and one matching effect vector")
    baseline = x_control.mean(axis=0)
    count = n or len(x_control)
    return BaselineResult("gene_effect_transfer",
                         np.repeat((baseline + effect)[None, :], count, axis=0),
                         {"source": "development_effect_matrix"})


def pert2state_baseline(prediction: np.ndarray, n: int | None = None) -> BaselineResult:
    """Register predictions produced by the pinned ``pert2state-model`` adapter.

    The orchestration environment intentionally does not import that optional
    research package.  Its isolated adapter writes a prediction array, which
    is validated here and then evaluated under the same frozen split as every
    other baseline.
    """
    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise ValueError("pert2state predictions must be a finite 2D array")
    if n is not None:
        if int(n) < 0:
            raise ValueError("prediction count cannot be negative")
        if len(prediction) < int(n):
            raise ValueError("pert2state predictions are shorter than the evaluation set")
        prediction = prediction[:int(n)]
    return BaselineResult("pert2state", prediction,
                          {"source": "pinned_pert2state_model_adapter", "adapter_only": True})


def ridge_residual(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> BaselineResult:
    """Small linear residual baseline with an explicit intercept."""
    x, y = map(lambda z: np.asarray(z, dtype=float), (x, y))
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("ridge inputs must be equal-length matrices")
    x1 = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(x1.shape[1]); reg[0, 0] = 0.0
    coef = np.linalg.solve(x1.T @ x1 + alpha * reg, x1.T @ y)
    return BaselineResult("linear_residual_ridge", x1 @ coef, {"alpha": alpha, "coef_shape": list(coef.shape)})


def fit_baseline_suite(x_control: np.ndarray, y_perturbed: np.ndarray, alpha: float = 1.0,
                       gene_effect: np.ndarray | None = None,
                       pert2state_prediction: np.ndarray | None = None) -> dict[str, BaselineResult]:
    """Return deterministic single-step baseline predictions."""
    result = {"condition_mean_effect": condition_mean_effect(x_control, y_perturbed),
              "linear_residual_ridge": ridge_residual(x_control, y_perturbed, alpha=alpha)}
    result["ntc_no_change"] = ntc_no_change(x_control, n=len(y_perturbed))
    if gene_effect is not None:
        result["gene_effect_transfer"] = gene_effect_transfer(x_control, gene_effect, n=len(y_perturbed))
    if pert2state_prediction is not None:
        result["pert2state"] = pert2state_baseline(pert2state_prediction, n=len(y_perturbed))
    return result
