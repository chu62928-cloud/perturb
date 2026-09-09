from __future__ import annotations

"""STATE/Cell-Eval compatible helpers for the D2 re-analysis.

The original D2 evaluation intentionally remains untouched.  This module
contains only the new, versioned evaluation primitives so that the historical
custom metrics and the STATE-compatible metrics cannot be mixed accidentally.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


PAPER_CORE_METRICS = (
    "pds_l1_paper",
    "pearson_delta",
    "pr_auc",
    "de_spearman_lfc_sig",
    "overlap_at_N",
    "de_spearman_sig",
)

CELL_EVAL_FULL_SKIP = ("pearson_edistance", "clustering_agreement")


def _as_float_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return array


def _check_bulk_inputs(
    real_pert: np.ndarray,
    pred_pert: np.ndarray,
    real_ctrl: np.ndarray,
    pred_ctrl: np.ndarray,
    gene_order: Sequence[str],
    perturbation_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    arrays = tuple(
        _as_float_matrix(value, name)
        for value, name in (
            (real_pert, "real_pert"),
            (pred_pert, "pred_pert"),
            (real_ctrl, "real_ctrl"),
            (pred_ctrl, "pred_ctrl"),
        )
    )
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError("all pseudobulk matrices must have the same shape")
    genes = [str(gene) for gene in gene_order]
    perts = [str(perturbation) for perturbation in perturbation_names]
    if first_shape != (len(perts), len(genes)):
        raise ValueError("pseudobulk shape does not match perturbation names or gene order")
    if len(set(perts)) != len(perts):
        raise ValueError("perturbation names must be unique within one context")
    return (*arrays, genes, perts)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with explicit NaN for a constant vector."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def paper_pearson_delta(
    real_pert: np.ndarray,
    pred_pert: np.ndarray,
    real_ctrl: np.ndarray,
    pred_ctrl: np.ndarray,
    gene_order: Sequence[str],
    perturbation_names: Sequence[str],
) -> dict[str, float]:
    """Compute signed Pearson Delta for each perturbation.

    This follows the operational implementation used by paper-era and current
    Cell-Eval: each perturbation is compared with its corresponding control
    pseudobulk, then Pearson correlation is computed across genes.  The caller
    performs the perturbation-level macro average.
    """
    real_pert, pred_pert, real_ctrl, pred_ctrl, _genes, perts = _check_bulk_inputs(
        real_pert, pred_pert, real_ctrl, pred_ctrl, gene_order, perturbation_names
    )
    real_delta = real_pert - real_ctrl
    pred_delta = pred_pert - pred_ctrl
    return {
        pert: _pearson(real_delta[index], pred_delta[index])
        for index, pert in enumerate(perts)
    }


def paper_pearson_delta_absolute_text(
    real_pert: np.ndarray,
    pred_pert: np.ndarray,
    real_ctrl: np.ndarray,
    pred_ctrl: np.ndarray,
    gene_order: Sequence[str],
    perturbation_names: Sequence[str],
) -> dict[str, float]:
    """Sensitivity variant matching the paper sentence describing absolute deltas."""
    real_pert, pred_pert, real_ctrl, pred_ctrl, _genes, perts = _check_bulk_inputs(
        real_pert, pred_pert, real_ctrl, pred_ctrl, gene_order, perturbation_names
    )
    real_delta = np.abs(real_pert - real_ctrl)
    pred_delta = np.abs(pred_pert - pred_ctrl)
    return {
        pert: _pearson(real_delta[index], pred_delta[index])
        for index, pert in enumerate(perts)
    }


def paper_pds_l1(
    real_pert: np.ndarray,
    pred_pert: np.ndarray,
    real_ctrl: np.ndarray,
    pred_ctrl: np.ndarray,
    gene_order: Sequence[str],
    perturbation_names: Sequence[str],
) -> dict[str, float]:
    """Paper-era PDS-L1 on absolute perturbation effects.

    The paper-era Cell-Eval implementation returns ``rank / T`` (lower is
    better).  STATE's reproduction notebook reports the transformed score
    ``1 - 2 * rank / T`` so that a random prediction is approximately zero and
    a perfect prediction is one.
    """
    real_pert, pred_pert, real_ctrl, pred_ctrl, genes, perts = _check_bulk_inputs(
        real_pert, pred_pert, real_ctrl, pred_ctrl, gene_order, perturbation_names
    )
    real_effects = np.abs(real_pert - real_ctrl)
    pred_effects = np.abs(pred_pert - pred_ctrl)
    n_perts = len(perts)
    output: dict[str, float] = {}
    for index, pert in enumerate(perts):
        include = np.asarray([gene != pert for gene in genes], dtype=bool)
        distances = np.abs(real_effects[:, include] - pred_effects[index, include]).sum(axis=1)
        order = np.argsort(distances, kind="stable")
        rank = int(np.flatnonzero(order == index)[0])
        output[pert] = float(1.0 - 2.0 * rank / max(n_perts, 1))
    return output


def macro_mean(values: Mapping[str, float], *, finite_only: bool = True) -> float:
    """Macro-average perturbation scores without converting NaN to zero."""
    array = np.asarray(list(values.values()), dtype=np.float64)
    if finite_only:
        array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(array.mean())


def validate_cell_eval_result_columns(columns: Iterable[str]) -> set[str]:
    """Return the metric columns emitted by Cell-Eval and reject malformed output."""
    names = {str(column) for column in columns}
    if "perturbation" not in names:
        raise ValueError("Cell-Eval results must contain a perturbation column")
    return names - {"perturbation"}


@dataclass(frozen=True)
class ContextMetricSummary:
    condition: str
    n_perturbations: int
    finite_counts: dict[str, int]
    macro_metrics: dict[str, float]


def summarize_context_metrics(
    condition: str,
    rows: Sequence[Mapping[str, object]],
    metric_names: Sequence[str],
) -> ContextMetricSummary:
    """Aggregate one condition after Cell-Eval has produced per-perturbation rows."""
    if not rows:
        raise ValueError("cannot summarize an empty condition")
    macro: dict[str, float] = {}
    finite_counts: dict[str, int] = {}
    for metric in metric_names:
        values = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                values.append(numeric)
        finite_counts[metric] = len(values)
        macro[metric] = float(np.mean(values)) if values else float("nan")
    return ContextMetricSummary(
        condition=str(condition),
        n_perturbations=len(rows),
        finite_counts=finite_counts,
        macro_metrics=macro,
    )
