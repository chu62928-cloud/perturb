from __future__ import annotations

import numpy as np


def support_score(predicted_latent: np.ndarray, train_latent: np.ndarray,
                  second_action_latent: np.ndarray | None = None, k: int = 20) -> dict:
    """Calculate auditable intermediate-state support signals."""
    predicted_latent = np.asarray(predicted_latent, dtype=float)
    train_latent = np.asarray(train_latent, dtype=float)
    if predicted_latent.ndim == 1:
        predicted_latent = predicted_latent[None, :]
    if train_latent.ndim != 2:
        raise ValueError("train_latent must be a matrix")
    distances = np.sqrt(((predicted_latent[:, None, :] - train_latent[None, :, :]) ** 2).sum(axis=2))
    nearest = distances.min(axis=1)
    kth = np.partition(distances, min(k, distances.shape[1] - 1), axis=1)[:, :k]
    density = 1.0 / (kth.mean(axis=1) + 1e-8)
    result = {"nearest_distance": float(nearest.mean()), "local_density": float(density.mean()),
              "effective_cells": int(min(k, train_latent.shape[0])),
              "n_predictions": int(predicted_latent.shape[0])}
    if second_action_latent is not None:
        result["second_action_support_distance"] = float(np.mean(np.sqrt(((predicted_latent[:, None, :] - np.asarray(second_action_latent)[None, :, :]) ** 2).sum(axis=2)).min(axis=1)))
    return result


def pass_support(observed: dict, calibration: dict) -> tuple[bool, list[str]]:
    reasons = []
    if observed.get("nearest_distance", float("inf")) > calibration["nearest_distance_p95"]:
        reasons.append("intermediate state is outside kNN distance support")
    if observed.get("local_density", 0.0) < calibration["density_p05"]:
        reasons.append("intermediate state has low local density")
    if observed.get("effective_cells", 0) < calibration["min_effective_cells"]:
        reasons.append("insufficient local cells for second action")
    return not reasons, reasons

