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
        groups = np.asarray([_text(x) for x in obj.obs["guide_group"].to_numpy()])
        quality = np.asarray([_bool(x) for x in obj.obs["low_quality"].to_numpy()])
        names = np.asarray([_text(x) for x in obj.obs["perturbed_gene_name"].to_numpy()])
        ids = np.asarray([_text(x) for x in obj.obs["perturbed_gene_id"].to_numpy()])
        types = np.asarray([_text(x) for x in obj.obs["guide_type"].to_numpy()])
        out = {}
        base = (groups == SINGLE_GUIDE_GROUP) & (~quality)
        ntc = base & np.asarray([bool(re.search(r"ntc|non[-_ ]?target|control|negative", x, re.I)) for x in types])
        if "NTC" in wanted:
            out["NTC"] = np.flatnonzero(ntc)[:max_rows]
        for gene in sorted(wanted - {"NTC"}):
            mask = base & ((names == gene) | (ids == gene))
            out[gene] = np.flatnonzero(mask)[:max_rows]
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
    with h5py.File(path, "r") as handle:
        order = np.argsort(rows)
        sorted_rows = rows[order]
        for start in range(0, len(sorted_rows), 512):
            chunk = sorted_rows[start:start + 512]
            first, last = int(chunk[0]), int(chunk[-1])
            block = _read_csr_block_columns_h5(handle, first, last + 1, columns, n_vars)
            pieces.append(block[chunk - first])
    values = np.vstack(pieces).astype(np.float32, copy=False)
    restored = np.empty_like(values)
    restored[order] = values
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

