from __future__ import annotations

"""Bounded D2 cell sampling for the STATE pilot and later training runs."""

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import _read_csr_block_columns_h5
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


def read_log_expression(path: str | Path, rows: Sequence[int], panel: Sequence[str]) -> np.ndarray:
    """Read selected rows and panel columns, recomputing CP10K from raw X."""
    import anndata as ad
    path = Path(path)
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.zeros((0, len(panel)), dtype=np.float32)
    columns = _panel_columns(path, panel)
    obj = ad.read_h5ad(path, backed="r")
    try:
        n_vars = int(obj.n_vars)
    finally:
        if getattr(obj, "file", None) is not None:
            obj.file.close()
    import h5py
    pieces = []
    sorted_values = []
    with h5py.File(path, "r") as handle:
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
            block = _read_csr_block_columns_h5(handle, first, last + 1, columns, n_vars)
            pieces.append((order[cursor:end], block[chunk - first]))
            sorted_values.append(block[chunk - first])
            cursor = end
    values = np.vstack(sorted_values).astype(np.float32, copy=False)
    restored = np.empty_like(values)
    for positions, piece in pieces:
        restored[positions] = piece
    totals = restored.sum(axis=1)
    totals = np.where(totals > 0, totals, 1.0)
    return np.log1p(restored / totals[:, None] * 10000.0).astype(np.float32)


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
                 device: str | None = None):
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

    def __iter__(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("D2 training stream requires the isolated PyTorch environment") from exc
        rng = np.random.default_rng(self.seed)
        while True:
            choices = rng.integers(0, len(self.records), size=self.batch_size)
            expressions = np.zeros((self.batch_size, self.set_len, len(self.panel)), dtype=np.float32)
            targets = np.zeros_like(expressions)
            perturbations = np.zeros((self.batch_size, self.set_len, len(self.names)), dtype=np.float32)
            grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
            for batch_index, choice in enumerate(choices):
                gene, condition = self.records[int(choice)]
                target_pool = self._rows[(gene, condition)]
                control_pool = self._rows[("NTC", condition)]
                target_rows = rng.choice(target_pool, size=self.set_len, replace=False)
                control_rows = rng.choice(control_pool, size=self.set_len, replace=False)
                grouped.setdefault(condition, []).append((batch_index, control_rows, target_rows))
                perturbations[batch_index, :, self.name_to_index[gene]] = 1.0
            for condition, entries in grouped.items():
                path = self._paths_by_condition[condition]
                controls = np.concatenate([entry[1] for entry in entries])
                responses = np.concatenate([entry[2] for entry in entries])
                control_values = read_log_expression(path, controls, self.panel)
                response_values = read_log_expression(path, responses, self.panel)
                for local, (batch_index, _, _) in enumerate(entries):
                    left, right = local * self.set_len, (local + 1) * self.set_len
                    expressions[batch_index] = control_values[left:right]
                    targets[batch_index] = response_values[left:right]
            yield {"expression": torch.as_tensor(expressions, device=self.device),
                   "perturbation": torch.as_tensor(perturbations, device=self.device),
                   "target": torch.as_tensor(targets, device=self.device),
                   "split": self.split, "d2_responses_used": True}
