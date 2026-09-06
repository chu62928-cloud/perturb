from __future__ import annotations

"""Bounded D2 cell sampling for the STATE pilot and later training runs."""

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import _read_indptr_slice_h5
from .guide_correction import SINGLE_GUIDE_GROUP, _text


def _bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "t", "yes", "y"}


def _condition(path: str | Path) -> str:
    match = re.search(r"_(Rest|Stim8hr|Stim48hr)(?:\.|$)", Path(path).name)
    if not match:
        raise ValueError(f"cannot infer D2 condition: {path}")
    return match.group(1)


def _equals(series, value: str) -> np.ndarray:
    """Compare a categorical obs column without decoding millions of strings."""
    if hasattr(series.dtype, "categories"):
        categories = np.asarray([_text(x) for x in series.cat.categories])
        hits = np.flatnonzero(categories == str(value))
        return np.isin(series.cat.codes.to_numpy(), hits)
    return np.asarray([_text(x) == str(value) for x in series.to_numpy()])


def _regex_categories(series, pattern: str) -> np.ndarray:
    if hasattr(series.dtype, "categories"):
        categories = np.asarray([_text(x) for x in series.cat.categories])
        hits = np.flatnonzero([bool(re.search(pattern, x, re.I)) for x in categories])
        return np.isin(series.cat.codes.to_numpy(), hits)
    return np.asarray([bool(re.search(pattern, _text(x), re.I)) for x in series.to_numpy()])


def _values_and_codes(series):
    if hasattr(series.dtype, "categories"):
        return np.asarray([_text(x) for x in series.cat.categories]), series.cat.codes.to_numpy()
    return None, np.asarray([_text(x) for x in series.to_numpy()])


def _panel_columns(path: Path, panel: Sequence[str]) -> list[int]:
    import anndata as ad
    obj = ad.read_h5ad(path, backed="r")
    try:
        ids = [_text(x) for x in obj.var["gene_ids"].tolist()]
        symbols = [_text(x) for x in obj.var["gene_name"].tolist()]
        by_symbol = {symbol: i for i, symbol in enumerate(symbols)}
        missing = [gene for gene in panel if gene not in by_symbol]
        if missing:
            raise ValueError(f"panel genes absent from {path.name}: {missing[:5]}")
        return [by_symbol[gene] for gene in panel]
    finally:
        if getattr(obj, "file", None) is not None:
            obj.file.close()


def eligible_rows_by_gene(path: str | Path, genes: Iterable[str], *, max_rows: int = 4096) -> dict[str, np.ndarray]:
    """Return at most ``max_rows`` deterministic eligible rows per gene."""
    import anndata as ad
    path = Path(path)
    wanted = {str(g) for g in genes}
    obj = ad.read_h5ad(path, backed="r")
    try:
        columns = ["guide_group", "low_quality", "perturbed_gene_name", "perturbed_gene_id", "guide_type"]
        missing = sorted(set(columns).difference(obj.obs.columns))
        if missing:
            raise ValueError(f"{path.name} missing pilot columns: {missing}")
        out = {}
        base = _equals(obj.obs["guide_group"], SINGLE_GUIDE_GROUP)
        low_quality = obj.obs["low_quality"].to_numpy()
        if low_quality.dtype != bool:
            low_quality = np.asarray([_bool(x) for x in low_quality])
        base &= ~low_quality
        ntc = base & _regex_categories(obj.obs["guide_type"], r"ntc|non[-_ ]?target|control|negative")
        if "NTC" in wanted:
            out["NTC"] = np.flatnonzero(ntc)[:max_rows]
        # Group once over the eligible rows.  Calling np.isin once per gene is
        # quadratic for the full D2 vocabulary (~12k targets).
        wanted_targets = wanted - {"NTC"}
        name_values, name_codes = _values_and_codes(obj.obs["perturbed_gene_name"])
        id_values, id_codes = _values_and_codes(obj.obs["perturbed_gene_id"])
        buckets = {gene: [] for gene in wanted_targets}
        if name_values is not None and id_values is not None:
            # D2 stores these columns categorically.  Resolve category codes
            # in vectorized form, then take only the first bounded rows per
            # target; this avoids an 8-million-row Python string loop.
            code_to_gene = {int(code): str(gene) for code, gene in enumerate(name_values)
                            if str(gene) in wanted_targets}
            name_hits = np.flatnonzero(base & np.isin(name_codes, list(code_to_gene)))
            target_codes = np.full(len(base), -1, dtype=np.int64)
            if len(name_hits):
                target_codes[name_hits] = name_codes[name_hits]
            id_offset = len(name_values)
            id_to_gene = {id_offset + int(code): str(gene) for code, gene in enumerate(id_values)
                          if str(gene) in wanted_targets}
            id_hits = np.flatnonzero(base & (target_codes < 0) &
                                     np.isin(id_codes, [code - id_offset for code in id_to_gene]))
            if len(id_hits):
                target_codes[id_hits] = id_offset + id_codes[id_hits]
                code_to_gene.update(id_to_gene)
            eligible = np.flatnonzero(target_codes >= 0)
            if len(eligible):
                order = np.argsort(target_codes[eligible], kind="stable")
                sorted_rows, sorted_codes = eligible[order], target_codes[eligible][order]
                boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1, len(sorted_codes)]
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    gene = code_to_gene.get(int(sorted_codes[left]))
                    if gene in buckets:
                        buckets[gene] = sorted_rows[left:min(right, left + max_rows)].astype(np.int64).tolist()
        else:
            for row in np.flatnonzero(base):
                name = _text(name_codes[row]) if name_values is None else ""
                gene_id = _text(id_codes[row]) if id_values is None else ""
                target = name or gene_id
                if target in buckets and len(buckets[target]) < max_rows:
                    buckets[target].append(int(row))
        out.update({gene: np.asarray(rows, dtype=np.int64) for gene, rows in buckets.items()})
        return out
    finally:
        if getattr(obj, "file", None) is not None:
            obj.file.close()


def eligible_counts_by_gene(path: str | Path, genes: Iterable[str]) -> dict[str, int]:
    """Count quality-filtered single-guide cells without reading expression values."""
    import anndata as ad
    path = Path(path)
    wanted = {str(g) for g in genes} - {"NTC"}
    obj = ad.read_h5ad(path, backed="r")
    try:
        base = _equals(obj.obs["guide_group"], SINGLE_GUIDE_GROUP)
        low_quality = obj.obs["low_quality"].to_numpy()
        if low_quality.dtype != bool:
            low_quality = np.asarray([_bool(x) for x in low_quality])
        base &= ~low_quality
        counts = {gene: 0 for gene in wanted}
        claimed = np.zeros(len(base), dtype=bool)
        for column in ("perturbed_gene_name", "perturbed_gene_id"):
            values, codes = _values_and_codes(obj.obs[column])
            if values is None:
                for row in np.flatnonzero(base & ~claimed):
                    gene = _text(codes[row])
                    if gene in counts:
                        counts[gene] += 1
                        claimed[row] = True
                continue
            code_to_gene = {int(code): str(gene) for code, gene in enumerate(values)
                            if str(gene) in wanted}
            if not code_to_gene:
                continue
            hits = base & ~claimed & np.isin(codes, list(code_to_gene))
            hit_codes, hit_counts = np.unique(codes[hits], return_counts=True)
            for code, count in zip(hit_codes, hit_counts):
                counts[code_to_gene[int(code)]] += int(count)
            claimed |= hits
        return counts
    finally:
        if getattr(obj, "file", None) is not None:
            obj.file.close()


def select_pilot_perturbations(counts_by_condition: Mapping[str, Mapping[str, int]],
                               splits: Mapping, perturbation_names: Sequence[str],
                               priority: Sequence[str], *, n_targets: int = 64,
                               set_len: int = 32, min_training_backgrounds: int = 2) -> dict:
    """Freeze a deterministic, response-blind pilot list from training support only."""
    names = [str(name) for name in perturbation_names if str(name) != "NTC"]
    if len(set(names)) != len(names) or n_targets <= 0:
        raise ValueError("pilot vocabulary and target count must be valid")
    train_pairs = {(str(row.get("perturbation_name")), str(row.get("condition")))
                   for row in splits.get("records", []) if row.get("split") == "train"}
    eligible = {}
    for gene in names:
        supported = {condition: int(counts.get(gene, 0))
                     for condition, counts in counts_by_condition.items()
                     if (gene, str(condition)) in train_pairs and int(counts.get(gene, 0)) >= set_len}
        if len(supported) >= min_training_backgrounds:
            eligible[gene] = {
                "training_cell_counts": supported,
                "minimum_training_cell_count": min(supported.values()),
                "training_background_count": len(supported),
            }
    missing_priority = [str(gene) for gene in priority if str(gene) not in eligible]
    if missing_priority:
        raise ValueError(f"priority pilot perturbations lack training support: {missing_priority}")
    selected = list(dict.fromkeys(map(str, priority)))
    ranked = sorted((gene for gene in eligible if gene not in selected),
                    key=lambda gene: (-eligible[gene]["minimum_training_cell_count"], gene))
    selected.extend(ranked[:max(0, n_targets - len(selected))])
    if len(selected) != n_targets:
        raise ValueError(f"only {len(selected)} perturbations meet the {n_targets}-target pilot rule")
    payload = {
        "version": "d2_state_pilot_manifest.v1",
        "selected_perturbations": selected,
        "priority_perturbations": list(map(str, priority)),
        "selection": {gene: eligible[gene] for gene in selected},
        "criteria": {"n_targets": n_targets, "set_len": set_len,
                     "min_training_backgrounds": min_training_backgrounds,
                     "ranking": "minimum_training_cell_count_desc_then_gene"},
        "split_hash": splits.get("split_hash"),
        "test_responses_used": False,
    }
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def build_d2_pilot_manifest(paths: Iterable[str | Path], perturbation_names: Sequence[str],
                            splits: Mapping, priority: Sequence[str], *, n_targets: int = 64,
                            set_len: int = 32) -> dict:
    paths = [Path(path) for path in paths]
    counts = {_condition(path): eligible_counts_by_gene(path, perturbation_names) for path in paths}
    return select_pilot_perturbations(counts, splits, perturbation_names, priority,
                                      n_targets=n_targets, set_len=set_len)


def _read_csr_panel_with_totals_h5(handle, start: int, stop: int,
                                    columns: Sequence[int], n_vars: int) -> tuple[np.ndarray, np.ndarray]:
    """Read panel counts and recomputed all-gene library sizes from raw X.

    STATE's CP10K normalization uses each cell's total count over the full
    measured gene axis.  The model input is then restricted to the frozen
    2,000-gene panel; using the panel sum as the denominator would make the
    normalization depend on the benchmark feature selection.
    """
    import h5py
    from scipy.sparse import csr_matrix

    x = handle["X"]
    if isinstance(x, h5py.Dataset):
        full = np.asarray(x[start:stop, :], dtype=np.float32)
        return full[:, columns].astype(np.float32, copy=False), full.sum(axis=1, dtype=np.float64).astype(np.float32)
    indptr = _read_indptr_slice_h5(handle, start, stop + 1)
    if len(indptr) and (indptr[0] < 0 or np.any(np.diff(indptr) < 0)):
        raise ValueError(f"invalid CSR indptr in rows {start}:{stop}; input file is corrupt")
    raw_start, raw_stop = int(indptr[0]), int(indptr[-1])
    indices = np.asarray(x["indices"][raw_start:raw_stop])
    values = np.asarray(x["data"][raw_start:raw_stop], dtype=np.float32)
    indptr -= raw_start
    matrix = csr_matrix((values, indices, indptr), shape=(stop - start, n_vars))
    panel = matrix[:, columns].toarray().astype(np.float32, copy=False)
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    return panel, totals


def _normalize_panel_counts(panel_counts: np.ndarray, all_gene_totals: np.ndarray) -> np.ndarray:
    """Apply the frozen CP10K followed by log1p transform."""
    panel_counts = np.asarray(panel_counts, dtype=np.float32)
    all_gene_totals = np.asarray(all_gene_totals, dtype=np.float32)
    if panel_counts.ndim != 2 or all_gene_totals.shape != (panel_counts.shape[0],):
        raise ValueError("panel counts and all-gene totals have incompatible shapes")
    totals = np.where(all_gene_totals > 0, all_gene_totals, 1.0)
    return np.log1p(panel_counts / totals[:, None] * 10000.0).astype(np.float32)


def _read_log_expression_handle(handle, rows: Sequence[int], columns: Sequence[int],
                                 n_vars: int, panel_size: int) -> np.ndarray:
    """Read selected rows from an already-open CSR HDF5 handle."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.zeros((0, panel_size), dtype=np.float32)
    pieces = []
    sorted_values = []
    order = np.argsort(rows)
    sorted_rows = rows[order]
    cursor = 0
    while cursor < len(sorted_rows):
        end = cursor + 1
        while (end < len(sorted_rows) and end - cursor < 512 and
               int(sorted_rows[end] - sorted_rows[end - 1]) <= 64):
            end += 1
        chunk = sorted_rows[cursor:end]
        first, last = int(chunk[0]), int(chunk[-1])
        block, totals = _read_csr_panel_with_totals_h5(handle, first, last + 1, columns, n_vars)
        pieces.append((order[cursor:end], block[chunk - first], totals[chunk - first]))
        sorted_values.append((block[chunk - first], totals[chunk - first]))
        cursor = end
    values = np.vstack([item[0] for item in sorted_values]).astype(np.float32, copy=False)
    all_gene_totals = np.concatenate([item[1] for item in sorted_values]).astype(np.float32, copy=False)
    restored = np.empty_like(values)
    restored_totals = np.empty_like(all_gene_totals)
    for positions, piece, piece_totals in pieces:
        restored[positions] = piece
        restored_totals[positions] = piece_totals
    return _normalize_panel_counts(restored, restored_totals)


def read_log_expression(path: str | Path, rows: Sequence[int], panel: Sequence[str]) -> np.ndarray:
    """Read selected rows and panel columns, recomputing CP10K from raw X."""
    import anndata as ad
    import h5py
    path = Path(path)
    columns = _panel_columns(path, panel)
    obj = ad.read_h5ad(path, backed="r")
    try:
        n_vars = int(obj.n_vars)
    finally:
        if getattr(obj, "file", None) is not None:
            obj.file.close()
    with h5py.File(path, "r") as handle:
        return _read_log_expression_handle(handle, rows, columns, n_vars, len(panel))


def make_pilot_batch(paths: Iterable[str | Path], panel: Sequence[str], perturbations: Sequence[str],
                     condition: str = "Rest", set_len: int = 32, seed: int = 20260901) -> dict:
    """Build one real D2 target/control set for a pilot hard-contract check."""
    paths = [Path(path) for path in paths]
    path = next((item for item in paths if _condition(item) == condition), None)
    if path is None:
        raise ValueError(f"D2 condition not found: {condition}")
    rng = np.random.default_rng(seed)
    wanted = list(perturbations) + ["NTC"]
    rows = eligible_rows_by_gene(path, wanted, max_rows=max(set_len * 4, 128))
    missing = [gene for gene in perturbations if len(rows.get(gene, ())) < set_len]
    if missing:
        raise ValueError(f"pilot targets do not have {set_len} eligible cells in {condition}: {missing}")
    if len(rows.get("NTC", ())) < set_len:
        raise ValueError(f"NTC has fewer than {set_len} eligible cells in {condition}")
    target_gene = str(perturbations[0])
    target_rows = rng.choice(rows[target_gene], size=set_len, replace=False)
    control_rows = rng.choice(rows["NTC"], size=set_len, replace=False)
    target = read_log_expression(path, target_rows, panel)
    control = read_log_expression(path, control_rows, panel)
    return {"condition": condition, "target_gene": target_gene,
            "target_rows": target_rows.tolist(), "control_rows": control_rows.tolist(),
            "target_expression": target[None, :, :], "control_expression": control[None, :, :],
            "perturbation_names": list(perturbations), "d2_responses_used": True}


class D2BatchStream:
    """Re-iterable, bounded CSR stream for fair STATE train/validation batches.

    Each example pairs a perturbation response set with an independently
    sampled NTC set from the same biological background.  The NTC set is a
    population background region, not a claim about a cell's true pre-state.
    Only rows whose `(perturbation, condition)` record belongs to ``split``
    are sampled, so the same stream can be used for Scratch and Transfer.
    """

    def __init__(self, paths: Iterable[str | Path], panel: Sequence[str],
                 perturbation_names: Sequence[str], splits: Mapping, *,
                 split: str = "train", batch_size: int = 64, set_len: int = 32,
                 max_rows_per_gene: int = 128, seed: int = 20260901,
                 device: str | None = None, cache_expression: bool = True,
                 cache_max_bytes: int = 28 * 1024**3):
        if split not in {"train", "validation", "test"}:
            raise ValueError("D2 split must be train, validation or test")
        if batch_size <= 0 or set_len <= 0 or max_rows_per_gene < set_len:
            raise ValueError("batch_size, set_len and max_rows_per_gene are inconsistent")
        self.paths = [Path(path) for path in paths]
        self.panel = list(panel)
        self.names = list(map(str, perturbation_names))
        if not self.names or self.names[0] != "NTC" or len(set(self.names)) != len(self.names):
            raise ValueError("D2 vocabulary must start with unique NTC")
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.batch_size = int(batch_size)
        self.set_len = int(set_len)
        self.split = split
        self.seed = int(seed)
        self.device = device
        self.cache_expression = bool(cache_expression)
        self.cache_max_bytes = int(cache_max_bytes)
        if self.cache_max_bytes < 0:
            raise ValueError("cache_max_bytes must be non-negative")
        self._expression_cache: dict[tuple[str, str], np.ndarray] = {}
        self._expression_cache_bytes = 0
        records = [row for row in splits.get("records", []) if str(row.get("split")) == split]
        if not records:
            raise ValueError(f"D2 split has no {split} records")
        self._rows: dict[tuple[str, str], np.ndarray] = {}
        wanted = self.names
        for path in self.paths:
            condition = _condition(path)
            found = eligible_rows_by_gene(path, wanted, max_rows=max_rows_per_gene)
            for gene, rows in found.items():
                self._rows[(str(gene), condition)] = np.asarray(rows, dtype=np.int64)
        self.records = []
        for row in records:
            gene, condition = str(row.get("perturbation_name")), str(row.get("condition"))
            target_rows = self._rows.get((gene, condition), np.zeros(0, dtype=np.int64))
            control_rows = self._rows.get(("NTC", condition), np.zeros(0, dtype=np.int64))
            if len(target_rows) >= self.set_len and len(control_rows) >= self.set_len:
                if gene not in self.name_to_index:
                    raise ValueError(f"D2 record is absent from perturbation vocabulary: {gene}")
                self.records.append((gene, condition))
        if not self.records:
            raise ValueError(f"D2 {split} split has no records with {self.set_len}-cell support")
        self._paths_by_condition = {_condition(path): path for path in self.paths}
        self._panel_columns_by_condition = {
            condition: _panel_columns(path, self.panel)
            for condition, path in self._paths_by_condition.items()
        }
        self._n_vars_by_condition = {}
        self._h5_handles = {}
        import anndata as ad
        import h5py
        for condition, path in self._paths_by_condition.items():
            obj = ad.read_h5ad(path, backed="r")
            try:
                self._n_vars_by_condition[condition] = int(obj.n_vars)
            finally:
                if getattr(obj, "file", None) is not None:
                    obj.file.close()
            self._h5_handles[condition] = h5py.File(path, "r")

    def _draw_assignments(self, rng) -> list[tuple[int, str, str, np.ndarray, np.ndarray]]:
        choices = rng.integers(0, len(self.records), size=self.batch_size)
        assignments = []
        for batch_index, choice in enumerate(choices):
            gene, condition = self.records[int(choice)]
            target_pool = self._rows[(gene, condition)]
            control_pool = self._rows[("NTC", condition)]
            target_rows = rng.choice(target_pool, size=self.set_len, replace=False)
            control_rows = rng.choice(control_pool, size=self.set_len, replace=False)
            assignments.append((batch_index, gene, condition, control_rows, target_rows))
        return assignments

    def _expression_pool(self, gene: str, condition: str) -> tuple[np.ndarray, np.ndarray]:
        """Load one bounded target/control pool once and reuse it across steps."""
        key = (str(gene), str(condition))
        cached = self._expression_cache.get(key)
        if cached is not None:
            return self._rows[key], cached
        rows = self._rows[key]
        values = _read_log_expression_handle(
            self._h5_handles[condition], rows,
            self._panel_columns_by_condition[condition],
            self._n_vars_by_condition[condition], len(self.panel))
        if self.cache_expression and values.nbytes <= self.cache_max_bytes:
            while self._expression_cache and self._expression_cache_bytes + values.nbytes > self.cache_max_bytes:
                old_key = next(iter(self._expression_cache))
                old_value = self._expression_cache.pop(old_key)
                self._expression_cache_bytes -= old_value.nbytes
            if values.nbytes <= self.cache_max_bytes:
                self._expression_cache[key] = values
                self._expression_cache_bytes += values.nbytes
        return rows, values

    def iter_from(self, completed_batches: int = 0):
        """Yield the deterministic stream after cheaply replaying RNG draws."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("D2 training stream requires the isolated PyTorch environment") from exc
        if completed_batches < 0:
            raise ValueError("completed_batches must be non-negative")
        rng = np.random.default_rng(self.seed)
        for _ in range(int(completed_batches)):
            self._draw_assignments(rng)
        while True:
            expressions = np.zeros((self.batch_size, self.set_len, len(self.panel)), dtype=np.float32)
            targets = np.zeros_like(expressions)
            perturbations = np.zeros((self.batch_size, self.set_len, len(self.names)), dtype=np.float32)
            grouped: dict[str, list[tuple[int, str, np.ndarray, np.ndarray]]] = {}
            batch_genes = [""] * self.batch_size
            for batch_index, gene, condition, control_rows, target_rows in self._draw_assignments(rng):
                grouped.setdefault(condition, []).append((batch_index, gene, control_rows, target_rows))
                perturbations[batch_index, :, self.name_to_index[gene]] = 1.0
                batch_genes[batch_index] = gene
            for condition, entries in grouped.items():
                control_rows, control_values = self._expression_pool("NTC", condition)
                target_pools = {
                    gene: self._expression_pool(gene, condition)
                    for gene in {entry[1] for entry in entries}
                }
                for batch_index, gene, selected_controls, selected_targets in entries:
                    control_indices = np.searchsorted(control_rows, selected_controls)
                    target_rows, target_values = target_pools[gene]
                    target_indices = np.searchsorted(target_rows, selected_targets)
                    if (np.any(control_indices >= len(control_rows)) or
                            np.any(control_rows[control_indices] != selected_controls) or
                            np.any(target_indices >= len(target_rows)) or
                            np.any(target_rows[target_indices] != selected_targets)):
                        raise RuntimeError("D2 expression pool rows are not sorted or missing")
                    expressions[batch_index] = control_values[control_indices]
                    targets[batch_index] = target_values[target_indices]
            yield {"expression": torch.as_tensor(expressions, device=self.device),
                   "perturbation": torch.as_tensor(perturbations, device=self.device),
                   "target": torch.as_tensor(targets, device=self.device),
                   "perturbation_names": batch_genes,
                   "split": self.split, "d2_responses_used": True}

    def __iter__(self):
        return self.iter_from(0)

    def close(self):
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __del__(self):  # pragma: no cover - best-effort cleanup on process exit
        self.close()
