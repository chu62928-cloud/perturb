from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HoldoutSplit:
    name: str
    train: tuple
    test: tuple
    seed: int
    rule: str


@dataclass(frozen=True)
class StateRegion:
    condition: str
    anchor_index: int
    anchor: tuple[float, ...]
    radius: float
    fold: int
    density: float


@dataclass(frozen=True)
class ContinuousStateSplit:
    regions: tuple[StateRegion, ...]
    fold_masks: tuple[dict[str, np.ndarray], ...]
    dimensions: int
    k_neighbors: int
    buffer_scale: float
    state_unit: str = "continuous_knn_ball"
    leiden_role: str = "reporting_only"


def _stable_order(items: Iterable, seed: int) -> list:
    return sorted(items, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())


def gene_holdout(genes: Iterable[str], fraction: float = .20, seed: int = 20260901) -> HoldoutSplit:
    values = list(dict.fromkeys(map(str, genes)))
    ordered = _stable_order(values, seed)
    n_test = max(1, round(len(ordered) * fraction)) if ordered else 0
    test = tuple(sorted(ordered[:n_test])); train = tuple(sorted(ordered[n_test:]))
    return HoldoutSplit("gene", train, test, seed, "unseen genes; N/A for one-hot models")


def state_holdout(states: Iterable[str], seed: int = 20260901) -> HoldoutSplit:
    values = list(dict.fromkeys(map(str, states)))
    ordered = _stable_order(values, seed)
    test = tuple(sorted(ordered[:1])); train = tuple(sorted(ordered[1:]))
    return HoldoutSplit("state", train, test, seed, "leave-one-local-state-out")


def interaction_holdout(pairs: Iterable[tuple[str, str]], fraction: float = .20,
                        seed: int = 20260901) -> HoldoutSplit:
    values = list(dict.fromkeys((str(g), str(s)) for g, s in pairs))
    ordered = _stable_order(values, seed)
    n_test = max(1, round(len(ordered) * fraction)) if ordered else 0
    test = tuple(sorted(ordered[:n_test])); train = tuple(sorted(ordered[n_test:]))
    return HoldoutSplit("gene_x_state", train, test, seed, "both gene and state marginals remain observed")


def three_way_split(items: Iterable, validation_fraction: float = .20,
                    test_fraction: float = .20, seed: int = 20260901) -> dict[str, list]:
    """Deterministically split items into train/validation/test partitions."""
    values = list(dict.fromkeys(items))
    if not values or validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid three-way split fractions or empty items")
    ordered = _stable_order(values, seed)
    n_test = max(1, round(len(ordered) * test_fraction))
    n_validation = max(1, round(len(ordered) * validation_fraction))
    if n_test + n_validation >= len(ordered):
        n_validation = max(1, len(ordered) - n_test - 1)
    return {"train": sorted(ordered[n_test + n_validation:]),
            "validation": sorted(ordered[n_test:n_test + n_validation]),
            "test": sorted(ordered[:n_test])}


def _farthest_anchors(points: np.ndarray, n: int, seed: int) -> list[int]:
    if len(points) == 0:
        return []
    n = min(n, len(points))
    first = int(seed % len(points))
    chosen = [first]
    nearest = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, n):
        idx = int(np.argmax(nearest))
        chosen.append(idx)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[idx], axis=1))
    return chosen


def _kth_neighbor_distances(points: np.ndarray, n_neighbors: int, block_size: int = 512) -> np.ndarray:
    """Return each point's kth neighbour distance without an n×n allocation.

    This NumPy fallback keeps the continuous-state split usable in the small
    pipeline environment where scikit-learn is unavailable.  Distances are
    evaluated in bounded matrix blocks; the production environment uses the
    equivalent ``NearestNeighbors`` implementation.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=float)
    k = min(max(1, int(n_neighbors)), n)
    norms = np.einsum("ij,ij->i", points, points)
    output = np.empty(n, dtype=float)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        distances = (norms[start:stop, None] + norms[None, :]
                     - 2.0 * points[start:stop] @ points.T)
        distances = np.maximum(distances, 0.0)
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        output[start:stop] = np.sqrt(nearest.max(axis=1))
    return output


def _point_kth_neighbor_distance(points: np.ndarray, index: int, n_neighbors: int) -> float:
    points = np.asarray(points, dtype=float)
    norms = np.einsum("ij,ij->i", points, points)
    distances = np.maximum(norms[index] + norms - 2.0 * (points @ points[index]), 0.0)
    k = min(max(1, int(n_neighbors)), len(points))
    return float(np.sqrt(np.partition(distances, k - 1)[k - 1]))


def continuous_state_regions(latent: np.ndarray, conditions: Iterable[str], *, n_anchors: int = 10,
                             k_neighbors: int = 200, n_folds: int = 5, seed: int = 20260901,
                             density_quantile: float = .10, buffer_scale: float = 1.25) -> ContinuousStateSplit:
    """Create deterministic continuous state regions and buffered holdouts.

    Regions are balls around farthest-point anchors selected from dense NTC
    cells.  Every fold contains two anchors per condition when possible.  The
    returned masks are suitable for gene, state, and gene×state evaluation;
    Leiden labels are intentionally not consumed here.
    """
    x = np.asarray(latent, dtype=float)
    cond = np.asarray([str(v) for v in conditions])
    if x.ndim != 2 or len(x) != len(cond) or not len(x):
        raise ValueError("latent and conditions must describe the same non-empty matrix")
    if n_folds < 1 or n_anchors < 1:
        raise ValueError("n_anchors and n_folds must be positive")
    # Nearest-neighbour queries are O(n*k) in memory.  The previous
    # implementation constructed a dense n×n distance matrix and was not
    # usable for the real NTC sample (tens of thousands of cells).
    try:
        from sklearn.neighbors import NearestNeighbors
    except ModuleNotFoundError:  # pragma: no cover - exercised in minimal local env
        NearestNeighbors = None

    regions: list[StateRegion] = []
    for condition in sorted(set(cond)):
        idx = np.flatnonzero(cond == condition)
        pts = x[idx]
        kk = min(k_neighbors, max(1, len(idx) - 1))
        n_query = min(len(idx), kk + 1)
        if NearestNeighbors is None:
            kth = _kth_neighbor_distances(pts, n_query)
        else:
            nn = NearestNeighbors(n_neighbors=n_query, algorithm="auto")
            nn.fit(pts)
            distances, _ = nn.kneighbors(pts, return_distance=True)
            kth = distances[:, -1]
        local_density = 1.0 / (kth + 1e-8)
        eligible = np.flatnonzero(local_density >= np.quantile(local_density, density_quantile))
        anchors = _farthest_anchors(pts[eligible], n_anchors, seed + len(regions))
        for rank, local_anchor in enumerate(anchors):
            anchor_local = int(eligible[local_anchor])
            anchor_global = int(idx[anchor_local])
            if NearestNeighbors is None:
                radius = _point_kth_neighbor_distance(pts, anchor_local, n_query)
            else:
                radius = float(nn.kneighbors(pts[anchor_local:anchor_local + 1],
                                             n_neighbors=n_query,
                                             return_distance=True)[0][0, -1])
            regions.append(StateRegion(condition, anchor_global, tuple(x[anchor_global]),
                                       max(radius, 1e-8), rank % n_folds,
                                       float(local_density[anchor_local])))
    masks: list[dict[str, np.ndarray]] = []
    for fold in range(n_folds):
        test = np.zeros(len(x), dtype=bool)
        buffered = np.zeros(len(x), dtype=bool)
        candidates: list[tuple[float, int, int]] = []
        for region in regions:
            if region.fold != fold:
                continue
            same = cond == region.condition
            distance = np.linalg.norm(x - np.asarray(region.anchor), axis=1)
            for cell in np.flatnonzero(same & (distance <= region.radius)):
                candidates.append((float(distance[cell] / region.radius), int(cell), region.fold))
            buffered |= same & (distance <= buffer_scale * region.radius)
        # Overlapping balls are resolved by nearest normalized anchor.  This
        # makes the fold masks disjoint while retaining the geometric buffer.
        for _, cell, _ in sorted(candidates):
            if not any(other["test"][cell] for other in masks):
                test[cell] = True
        masks.append({"test": test, "train": ~buffered, "buffer": buffered,
                      "fold": np.full(len(x), fold, dtype=int)})
    return ContinuousStateSplit(tuple(regions), tuple(masks), x.shape[1],
                                k_neighbors, buffer_scale)


def state_split_payload(split: ContinuousStateSplit, latent: np.ndarray | None = None,
                        conditions: Iterable[str] | None = None) -> dict:
    """Serialize state regions, memberships and a reproducibility hash."""
    payload = {
        "version": "continuous_state_split.v1",
        "state_unit": split.state_unit,
        "leiden_role": split.leiden_role,
        "dimensions": split.dimensions,
        "k_neighbors": split.k_neighbors,
        "buffer_scale": split.buffer_scale,
        "regions": [{"condition": r.condition, "anchor_index": r.anchor_index,
                     "anchor": list(r.anchor), "radius": r.radius,
                     "fold": r.fold, "density": r.density} for r in split.regions],
        "folds": [],
    }
    for masks in split.fold_masks:
        payload["folds"].append({"fold": int(masks["fold"][0]) if len(masks["fold"]) else 0,
                                 "test_cells": np.flatnonzero(masks["test"]).astype(int).tolist(),
                                 "buffer_cells": np.flatnonzero(masks["buffer"]).astype(int).tolist(),
                                 "train_cells": np.flatnonzero(masks["train"]).astype(int).tolist()})
    if conditions is not None:
        payload["condition_count"] = len(list(conditions))
    import json
    payload["split_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def validate_no_region_overlap(split: ContinuousStateSplit) -> bool:
    """Check test regions are disjoint across folds and have train buffers."""
    seen = np.zeros(len(split.fold_masks[0]["test"]), dtype=int)
    for masks in split.fold_masks:
        seen += masks["test"].astype(int)
        if np.any(masks["test"] & ~masks["buffer"]):
            return False
    return bool(np.all(seen <= 1))
