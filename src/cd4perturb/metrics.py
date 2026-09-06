from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    rmse: float
    standardized_rmse: float
    mae: float
    direction_accuracy: float
    n: int


def summarize(y_true: np.ndarray, y_pred: np.ndarray, baseline_scale: np.ndarray | None = None) -> MetricSummary:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} != {y_pred.shape}")
    error = y_pred - y_true
    scale = np.asarray(baseline_scale if baseline_scale is not None else np.nanstd(y_true, axis=0), dtype=float)
    scale = np.where(scale > 1e-8, scale, 1.0)
    direction = np.sign(y_true) == np.sign(y_pred)
    return MetricSummary(float(np.sqrt(np.mean(error ** 2))),
                        float(np.sqrt(np.mean((error / scale) ** 2))),
                        float(np.mean(np.abs(error))), float(np.mean(direction)), int(y_true.size))


def bootstrap_rmse_improvement(y_true: np.ndarray, pred: np.ndarray, baseline: np.ndarray,
                               n_boot: int = 1000, seed: int = 20260901) -> dict:
    rng = np.random.default_rng(seed)
    y_true, pred, baseline = map(lambda x: np.asarray(x, dtype=float), (y_true, pred, baseline))
    if not (y_true.shape == pred.shape == baseline.shape):
        raise ValueError("bootstrap arrays must have equal shapes")
    n = y_true.shape[0]
    improvements = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        model_rmse = np.sqrt(np.mean((pred[idx] - y_true[idx]) ** 2))
        base_rmse = np.sqrt(np.mean((baseline[idx] - y_true[idx]) ** 2))
        improvements[i] = 1.0 - model_rmse / max(base_rmse, 1e-12)
    q = np.quantile(improvements, [0.025, 0.5, 0.975])
    return {"lower": float(q[0]), "median": float(q[1]), "upper": float(q[2]), "n_boot": n_boot}


def hierarchical_bootstrap(values: np.ndarray, groups: Sequence[np.ndarray],
                           statistic: Callable[[np.ndarray], float] = np.mean,
                           n_boot: int = 1000, seed: int = 20260901,
                           scope: str = "D1_technical_state_uncertainty") -> dict:
    """Resample nested groups without pretending D1 estimates donor variance.

    ``groups`` is ordered outer-to-inner, e.g. gene→guide→cell or
    gene→region→guide→cell.  The cell level represents measurement noise;
    callers must use donor→... only after external donors are available.
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    arrays = [np.asarray(g) for g in groups]
    if not arrays or any(len(g) != len(vals) for g in arrays):
        raise ValueError("values and every hierarchy level must have equal length")
    if any(len(np.unique(g)) == 0 for g in arrays):
        raise ValueError("hierarchy levels cannot be empty")
    rng = np.random.default_rng(seed)

    def resample(indices: np.ndarray, level: int) -> np.ndarray:
        if level == len(arrays):
            return rng.choice(indices, size=len(indices), replace=True)
        labels = np.unique(arrays[level][indices])
        sampled = rng.choice(labels, size=len(labels), replace=True)
        out = [resample(indices[arrays[level][indices] == label], level + 1)
               for label in sampled]
        return np.concatenate(out) if out else np.empty(0, dtype=int)

    estimates = np.empty(n_boot, dtype=float)
    indices = np.arange(len(vals))
    observed = float(statistic(vals))
    for i in range(n_boot):
        sample = resample(indices, 0)
        estimates[i] = float(statistic(vals[sample]))
    q = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {"estimate": observed, "lower": float(q[0]), "median": float(q[1]),
            "upper": float(q[2]), "n_boot": int(n_boot), "scope": scope,
            "levels": int(len(arrays))}


def distribution_metrics(real: np.ndarray, predicted: np.ndarray) -> dict:
    real, predicted = map(lambda x: np.asarray(x, dtype=float), (real, predicted))
    if real.ndim != 2 or predicted.ndim != 2 or real.shape[1] != predicted.shape[1]:
        raise ValueError("distribution arrays must be 2D with equal feature count")
    mean_gap = float(np.linalg.norm(real.mean(0) - predicted.mean(0)))
    var_gap = float(np.linalg.norm(real.var(0) - predicted.var(0)))
    # A deterministic sliced-Wasserstein approximation without sklearn dependency.
    rng = np.random.default_rng(0)
    projections = rng.normal(size=(min(32, real.shape[1]), real.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12
    n = min(len(real), len(predicted))
    sw = []
    for p in projections:
        a = np.sort(real[:n] @ p)
        b = np.sort(predicted[:n] @ p)
        sw.append(float(np.mean(np.abs(a - b))))
    return {"mean_gap": mean_gap, "variance_gap": var_gap, "sliced_wasserstein": float(np.mean(sw))}


def biological_gate(metrics: dict, gate: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.get("relative_rmse_improvement", -1.0) < gate["rmse_improvement"]:
        reasons.append("standardized RMSE improvement below threshold")
    if metrics.get("bootstrap_ci_lower", -1.0) <= gate["bootstrap_ci_lower"]:
        reasons.append("bootstrap confidence interval includes zero")
    if metrics.get("program_direction_drop", 0.0) > gate["program_direction_margin"]:
        reasons.append("program direction accuracy is inferior")
    if metrics.get("critical_program_error_ratio", 1.0) > 1.0 + gate["critical_program_tolerance"]:
        reasons.append("critical program error is worse than baseline")
    if metrics.get("max_condition_rmse_ratio", 1.0) > 1.0 + gate["condition_rmse_tolerance"]:
        reasons.append("condition-specific performance collapse")
    if metrics.get("ntc_zero_effect_error", 0.0) > metrics.get("ntc_null_p95", float("inf")):
        reasons.append("NTC zero-effect error exceeds null range")
    return not reasons, reasons


def model_global_gate(metrics: dict, gate: dict) -> tuple[bool, list[str]]:
    """Gate average single-step performance on primary_64 only."""
    reasons: list[str] = []
    if metrics.get("evaluation_scope") not in (None, "primary_64"):
        reasons.append("global gate must be evaluated on primary_64")
    if metrics.get("relative_standardized_rmse_improvement",
                   metrics.get("relative_rmse_improvement", -1.0)) < gate.get("rmse_improvement", .05):
        reasons.append("standardized RMSE improvement below threshold")
    if metrics.get("bootstrap_ci_lower", -1.0) <= gate.get("bootstrap_ci_lower", 0.0):
        reasons.append("D1 hierarchical bootstrap lower bound is not positive")
    if metrics.get("program_direction_drop", 0.0) > gate.get("program_direction_margin", .02):
        reasons.append("program direction accuracy is inferior")
    if metrics.get("critical_program_error_ratio", 1.0) > 1.0 + gate.get("critical_program_tolerance", .05):
        reasons.append("critical program error is worse than baseline")
    if metrics.get("max_condition_rmse_ratio", 1.0) > 1.0 + gate.get("condition_rmse_tolerance", .10):
        reasons.append("condition-specific performance collapse")
    if metrics.get("ntc_zero_effect_error", 0.0) > metrics.get("ntc_null_p95", float("inf")):
        reasons.append("NTC zero-effect error exceeds null range")
    return not reasons, reasons


def model_state_gate(metrics: dict, gate: dict) -> tuple[bool, list[str]]:
    """Gate state-dependent prediction before composition testing."""
    reasons: list[str] = []
    if metrics.get("evaluation_scope") not in (None, "primary_64"):
        reasons.append("state gate must be evaluated on primary_64")
    if metrics.get("relative_standardized_rmse_improvement",
                   metrics.get("relative_rmse_improvement", -1.0)) < gate.get("rmse_improvement", .05):
        reasons.append("joint gene×continuous-state RMSE improvement below threshold")
    if metrics.get("bootstrap_ci_lower", -1.0) <= gate.get("bootstrap_ci_lower", 0.0):
        reasons.append("state-level bootstrap lower bound is not positive")
    if metrics.get("local_direction_drop", metrics.get("program_direction_drop", 0.0)) > gate.get("program_direction_margin", .02):
        reasons.append("local perturbation direction accuracy is inferior")
    improved = int(metrics.get("distribution_metrics_improved", 0))
    if improved < int(gate.get("state_distribution_metrics_required", 2)):
        reasons.append("fewer than two distribution metrics improve over baseline")
    if metrics.get("state_proportion_error_ratio", 1.0) > 1.05:
        reasons.append("state proportion error is worse than baseline")
    if metrics.get("max_region_rmse_ratio", 1.0) > 1.10:
        reasons.append("a held-out continuous region has performance collapse")
    return not reasons, reasons
