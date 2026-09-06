from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CompositionMatch:
    """Sealed geometry-only intermediate-state matching manifest."""
    version: str
    rows: tuple[dict, ...]
    geometry_hash: str
    sealed: bool = False
    second_step_accessed: bool = False


# Descriptive aliases used by workflow rules and audit reports.
MatchManifest = CompositionMatch


def _hash_rows(rows: Iterable[dict]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def match_intermediate_states(predicted_latent: np.ndarray, real_latent: np.ndarray,
                              conditions: Iterable[str], calibration: dict,
                              second_action_support: dict[str, int] | None = None) -> CompositionMatch:
    """Match predicted states by geometry only; no second-step outcomes."""
    pred = np.asarray(predicted_latent, dtype=float)
    real = np.asarray(real_latent, dtype=float)
    cond = np.asarray([str(x) for x in conditions])
    if pred.ndim != 2 or real.ndim != 2 or pred.shape[1] != real.shape[1] or len(real) != len(cond):
        raise ValueError("latent matrices and conditions have incompatible shapes")
    max_distance = float(calibration["max_distance"])
    min_density = float(calibration.get("min_density", 0.0))
    min_cells = int(calibration.get("min_effective_cells", 20))
    rows = []
    requested_condition = calibration.get("condition")
    for i, point in enumerate(pred):
        allowed = np.flatnonzero(cond == str(requested_condition)) if requested_condition else np.arange(len(real))
        if not len(allowed):
            allowed = np.arange(len(real))
        dist = np.linalg.norm(real[allowed] - point, axis=1)
        order = np.argsort(dist)
        nearest = int(allowed[order[0]])
        k = min(int(calibration.get("k", 200)), len(order))
        density = float(1.0 / (dist[order[:k]].mean() + 1e-8))
        n_support = int(second_action_support.get(str(i), min_cells)
                        if second_action_support else min_cells)
        rows.append({"prediction_index": i, "matched_index": nearest,
                     "distance": float(dist[order[0]]), "local_density": density,
                     "effective_cells": n_support, "matched_condition": str(cond[nearest]),
                     "matched": bool(dist[order[0]] <= max_distance and density >= min_density
                                      and n_support >= min_cells)})
    return CompositionMatch("composition_match_v1", tuple(rows), _hash_rows(rows))


def seal_match(manifest: CompositionMatch) -> CompositionMatch:
    return CompositionMatch(manifest.version, manifest.rows, manifest.geometry_hash, True,
                            manifest.second_step_accessed)


seal_match_manifest = seal_match
match_composition = match_intermediate_states


def score_composition(manifest: CompositionMatch, second_step_results: dict) -> dict:
    """Score outcomes only after the geometry manifest has been sealed."""
    if not manifest.sealed:
        raise RuntimeError("match manifest must be sealed before second-step outcomes are opened")
    if manifest.second_step_accessed:
        raise RuntimeError("composition outcomes cannot be scored twice")
    expected = {int(row["prediction_index"]) for row in manifest.rows if row["matched"]}
    missing = expected.difference(map(int, second_step_results))
    if missing:
        raise ValueError(f"second-step outcomes missing matched indices: {sorted(missing)}")
    values = [float(second_step_results[i]) for i in sorted(expected)]
    return {"n_valid": len(values), "mean_second_step_score": float(np.mean(values)) if values else float("nan"),
            "manifest_hash": manifest.geometry_hash, "composition_stage": "score"}


def composition_gate(metrics: dict, error_amplification: float = .10,
                     direction_drop: float = .02, support_pass_rate: float = .80) -> tuple[bool, list[str]]:
    """Independent gate for proxy two-step composability."""
    reasons = []
    if "n_valid" in metrics and metrics.get("n_valid", 0) < metrics.get("min_valid_triplets", 50):
        reasons.append("fewer than 50 valid proxy triplets")
    if "n_second_actions" in metrics and metrics["n_second_actions"] < metrics.get("min_second_actions", 20):
        reasons.append("fewer than 20 second-action genes are covered")
    if "n_regions" in metrics and metrics["n_regions"] < metrics.get("min_regions", 3):
        reasons.append("fewer than three continuous state regions are covered")
    if metrics.get("second_step_rmse_ratio", float("inf")) > 1.0:
        reasons.append("second-step error is worse than composable baseline")
    if metrics.get("error_amplification", float("inf")) > error_amplification:
        reasons.append("error amplification exceeds threshold")
    if metrics.get("program_direction_drop", float("inf")) > direction_drop:
        reasons.append("program direction drop exceeds threshold")
    if metrics.get("support_pass_rate", 0.0) < support_pass_rate:
        reasons.append("intermediate support pass rate is too low")
    if metrics.get("distribution_collapse", False):
        reasons.append("distribution metrics collapsed")
    return not reasons, reasons
