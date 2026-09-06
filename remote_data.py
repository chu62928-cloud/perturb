from __future__ import annotations

"""Memory-bounded readers for the large assigned-guide AnnData files."""

from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .guide_correction import SINGLE_GUIDE_GROUP, _text, load_guide_library


def _as_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _sample_csr_columns(path: Path, rows: np.ndarray, columns: list[int]) -> np.ndarray:
    """Read selected CSR rows without relying on AnnData fancy indexing.

    AnnData 0.11 backed sparse indexing is not compatible with every recent
    SciPy release.  Direct HDF5 CSR slicing keeps this reader stable and only
    materializes ``len(rows) × len(columns)`` values.
    """
    import h5py

    rows = np.asarray(rows, dtype=np.int64)
    columns = list(map(int, columns))
    lookup = {column: j for j, column in enumerate(columns)}
    result = np.zeros((len(rows), len(columns)), dtype=np.float32)
    with h5py.File(path, "r") as handle:
        indptr = handle["X/indptr"]
        indices = handle["X/indices"]
        data = handle["X/data"]
        # Read contiguous row windows.  This avoids thousands of tiny HDF5
        # reads and works around B-tree/cache issues seen with large CSR files.
        order = np.argsort(rows)
        sorted_rows = rows[order]
        for offset in range(0, len(sorted_rows), 512):
            window_rows = sorted_rows[offset:offset + 512]
            first, last = int(window_rows[0]), int(window_rows[-1])
            pointers = np.asarray(indptr[first:last + 2])
            raw_start, raw_stop = int(pointers[0]), int(pointers[-1])
            row_indices = np.asarray(indices[raw_start:raw_stop])
            row_values = np.asarray(data[raw_start:raw_stop])
            for source_row in window_rows:
                local = int(source_row) - first
                start = int(pointers[local] - raw_start)
                stop = int(pointers[local + 1] - raw_start)
                out_row = int(order[offset + np.where(window_rows == source_row)[0][0]])
                for column, value in zip(row_indices[start:stop], row_values[start:stop]):
                    target = lookup.get(int(column))
                    if target is not None:
                        result[out_row, target] = value
    return result


def _sequential_sample_columns(obj, rows: np.ndarray, columns: list[int], max_gap: int = 64) -> np.ndarray:
    """Read selected rows using slice-only access and preserve their order."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if not len(rows):
        return np.zeros((0, len(columns)), dtype=np.float32)
    pieces: list[np.ndarray] = []
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and int(rows[end] - rows[end - 1]) <= max_gap:
            end += 1
        first, last = int(rows[start]), int(rows[end - 1])
        # Slice rows first; selecting columns on the resulting in-memory CSR
        # avoids backed combined fancy indexing.
        block = obj.X[first:last + 1, :][:, columns]
        local = rows[start:end] - first
        pieces.append(_as_dense(block[local, :]).astype(np.float32, copy=False))
        start = end
    return np.vstack(pieces)


def _read_csr_block_columns(path: Path, start: int, stop: int,
                            columns: list[int], n_vars: int) -> np.ndarray:
    """Read one contiguous CSR row block and project selected columns.

    The raw CSR arrays are materialized only for ``start:stop``.  SciPy does
    the column projection in compiled code, avoiding Python iteration over
    millions of non-zero entries while keeping the peak allocation bounded by
    the block size.
    """
    import h5py
    from scipy.sparse import csr_matrix

    with h5py.File(path, "r") as handle:
        indptr = np.asarray(handle["X/indptr"][start:stop + 1], dtype=np.int64)
        raw_start, raw_stop = int(indptr[0]), int(indptr[-1])
        indices = np.asarray(handle["X/indices"][raw_start:raw_stop])
        values = np.asarray(handle["X/data"][raw_start:raw_stop], dtype=np.float32)
    indptr -= raw_start
    matrix = csr_matrix((values, indices, indptr), shape=(stop - start, n_vars))
    return matrix[:, columns].toarray().astype(np.float32, copy=False)


def _h5_text(value) -> str:
    """Decode one HDF5 string value without materialising a string column."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return _safe_text(value)


def _h5_strings(dataset) -> list[str]:
    """Read a small HDF5 string dataset (categories or var identifiers)."""
    return [_h5_text(value) for value in dataset[()]]


def _h5_obs_field(handle, name: str):
    """Return ``(categories, codes_or_dataset)`` for one obs field.

    AnnData stores categoricals as ``categories`` plus an integer ``codes``
    dataset.  Keeping that representation intact is important here: decoding
    3 million guide labels into Python strings was the main avoidable memory
    cost in the original implementation.
    """
    node = handle["obs"][name]
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        return _h5_strings(node["categories"]), node["codes"]
    return None, node


def _read_csr_block_columns_h5(handle, start: int, stop: int,
                               columns: list[int], n_vars: int) -> np.ndarray:
    """Read/project one CSR row block using an already-open HDF5 handle."""
    import h5py  # imported lazily so metadata-only users need not depend on it
    from scipy.sparse import csr_matrix

    x = handle["X"]
    if isinstance(x, h5py.Dataset):
        return np.asarray(x[start:stop, columns], dtype=np.float32)
    indptr = np.asarray(x["indptr"][start:stop + 1], dtype=np.int64)
    if len(indptr) and (indptr[0] < 0 or np.any(np.diff(indptr) < 0)):
        raise ValueError(f"invalid CSR indptr in rows {start}:{stop}; input file is corrupt")
    raw_start, raw_stop = int(indptr[0]), int(indptr[-1])
    indices = np.asarray(x["indices"][raw_start:raw_stop])
    values = np.asarray(x["data"][raw_start:raw_stop], dtype=np.float32)
    indptr -= raw_start
    matrix = csr_matrix((values, indices, indptr), shape=(stop - start, n_vars))
    return matrix[:, columns].toarray().astype(np.float32, copy=False)


def _validate_csr_structure_h5(handle) -> None:
    """Fail cleanly on truncated/interior-zero CSR pointers.

    A malformed pointer array can make SciPy segfault while constructing a
    matrix.  Reading this small (about 25 MB for a 3-million-cell file)
    vector once is preferable to allowing a C-level crash half way through a
    long job.
    """
    import h5py

    x = handle["X"]
    if isinstance(x, h5py.Dataset):
        return
    pointers = np.asarray(x["indptr"][:], dtype=np.int64)
    if (not len(pointers) or pointers[0] != 0 or np.any(np.diff(pointers) < 0) or
            int(pointers[-1]) != int(x["data"].shape[0])):
        raise ValueError("invalid CSR indptr/data bounds; input H5AD is corrupt or truncated")


def _safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    return str(value)


def summarize_d1_candidates(paths: Iterable[str | Path], sample_size: int = 2048,
                            min_cells_per_guide: int = 100, min_guides_per_gene: int = 2,
                            seed: int = 20260901) -> list[dict]:
    """Build a target-direction-blind candidate table from D1 only.

    Only ``obs`` guide assignments and a small NTC/target expression sample are
    read.  No full matrix or D2 response is loaded.  The output intentionally
    leaves effect strata and functional annotations as ``unknown`` until the
    guide-library audit is supplied.
    """
    import anndata as ad

    rng = np.random.default_rng(seed)
    counts: dict[str, Counter] = defaultdict(Counter)
    cell_counts: Counter = Counter()
    expression_samples: dict[str, list[float]] = defaultdict(list)
    files = [Path(p) for p in paths]
    for path in files:
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs = obj.obs.copy()
            var_table = obj.var.copy()
            required = {"perturbed_gene_name", "guide_id", "guide_type", "low_quality"}
            missing = required.difference(obs.columns)
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            valid = ~obs["low_quality"].astype(bool)
            guide_type = obs["guide_type"].astype(str)
            gene_series = obs["perturbed_gene_name"].map(_safe_text)
            guide_series = obs["guide_id"].map(_safe_text)
            target_mask = valid & gene_series.ne("") & ~guide_type.str.contains("NTC|non-target|control", case=False, regex=True)
            for gene, guide in zip(gene_series[target_mask], guide_series[target_mask]):
                if guide:
                    counts[str(gene)][str(guide)] += 1
                    cell_counts[str(gene)] += 1

            ntc_mask = valid & guide_type.str.contains("NTC|non-target|control", case=False, regex=True)
            target_genes = [g for g, guides in counts.items() if len(guides) >= min_guides_per_gene]
            # Expression quartiles are a support signal, not a reason to read
            # every target column.  Cap the expensive sparse projection and
            # leave the remaining candidates explicitly unknown.
            target_genes = sorted(target_genes, key=lambda g: (-cell_counts[g], g))[:512]
            if target_genes and int(ntc_mask.sum()) > 0:
                ntc_idx = np.flatnonzero(ntc_mask.to_numpy())
                if len(ntc_idx) > sample_size:
                    # Evenly spaced sampling is deterministic and avoids
                    # random backed-row access on large CSR HDF5 datasets.
                    positions = np.linspace(0, len(ntc_idx) - 1, sample_size, dtype=int)
                    ntc_idx = ntc_idx[positions]
                var_names = [_safe_text(x) for x in obj.var_names]
                var_lookup = {name: i for i, name in enumerate(var_names)}
                if "gene_name" in var_table.columns:
                    for i, name in enumerate(var_table["gene_name"].map(_safe_text)):
                        if name and name not in var_lookup:
                            var_lookup[name] = i
                col_idx = [var_lookup[g] for g in target_genes if g in var_lookup]
                col_genes = [g for g in target_genes if g in var_lookup]
                if col_idx:
                    matrix = _sequential_sample_columns(obj, ntc_idx, col_idx)
                    for j, gene in enumerate(col_genes):
                        expression_samples[gene].extend(matrix[:, j].tolist())
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()

    candidates = []
    for gene, guides in counts.items():
        eligible_guides = {g: n for g, n in guides.items() if n >= min_cells_per_guide}
        if len(eligible_guides) < min_guides_per_gene:
            continue
        values = np.asarray(expression_samples.get(gene, []), dtype=float)
        median = float(np.median(values)) if values.size else None
        # Quantile boundaries are computed after aggregation below.
        candidates.append({"gene": gene, "guide_coverage": int(cell_counts[gene]),
                           "n_guides": int(len(eligible_guides)), "ntc_median_expression": median,
                           "effect_bin": "unknown", "guide_concordance": "unknown",
                           "functional_class": "unknown", "stimulus_dependent": False})
    if candidates:
        vals = np.asarray([x["ntc_median_expression"] for x in candidates if x["ntc_median_expression"] is not None], dtype=float)
        finite = vals[np.isfinite(vals)]
        q1, q3 = np.quantile(finite, [0.25, 0.75]) if finite.size else (0.0, 0.0)
        for row in candidates:
            value = row["ntc_median_expression"]
            row["expression_bin"] = "unknown" if value is None else "low" if value <= q1 else "high" if value >= q3 else "mid"
    return sorted(candidates, key=lambda x: (-x["guide_coverage"], x["gene"]))


def freeze_shared_gene_order(paths: Iterable[str | Path], pilot_genes: Iterable[str] = (),
                             n_genes: int = 2000, sample_size: int = 4096,
                             seed: int = 20260901, block_rows: int = 2048) -> dict:
    """Freeze a shared feature order from D1 NTC cells only.

    Gene IDs, rather than column positions, define the shared universe.  Each
    D1 file is aligned explicitly before statistics are combined, which is
    required because the 48-hour file has a leading custom feature.  The
    NTC sample is selected deterministically in row order and normalized with
    the observed ``total_counts`` field.  No perturbation response is read.
    """
    import anndata as ad

    files = [Path(p) for p in paths]
    if not files or any("D1" not in p.name for p in files):
        raise ValueError("freeze_shared_gene_order accepts D1 files only")
    tables: dict[str, dict] = {}
    for path in files:
        obj = ad.read_h5ad(path, backed="r")
        try:
            var = obj.var
            ids = [_safe_text(v) for v in var["gene_ids"]] if "gene_ids" in var else [_safe_text(v) for v in obj.var_names]
            names = [_safe_text(v) for v in var["gene_name"]] if "gene_name" in var else [_safe_text(v) for v in obj.var_names]
            feature = [_safe_text(v) for v in var["feature_types"]] if "feature_types" in var else ["Gene Expression"] * obj.n_vars
            mt = [bool(v) for v in var["mt"]] if "mt" in var else [n.upper().startswith("MT-") for n in names]
            by_id = {}
            for i, gene_id in enumerate(ids):
                if gene_id and gene_id not in by_id:
                    by_id[gene_id] = {"index": i, "gene_name": names[i],
                                      "feature_type": feature[i], "mt": mt[i]}
            tables[path.stem.split(".")[0]] = {"path": path, "obj": obj, "by_id": by_id,
                                                "n_vars": obj.n_vars}
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
    conditions = sorted(tables)
    shared = set(tables[conditions[0]]["by_id"])
    for condition in conditions[1:]:
        shared &= set(tables[condition]["by_id"])
    eligible = []
    names_by_id = {}
    for gene_id in sorted(shared):
        rows = [tables[c]["by_id"][gene_id] for c in conditions]
        if any(r["mt"] or r["feature_type"] != "Gene Expression" or gene_id.upper().startswith("CUSTOM")
               for r in rows):
            continue
        name = next((r["gene_name"] for r in rows if r["gene_name"]), gene_id)
        if name.upper().startswith("MT-") or name.upper().startswith("CUSTOM"):
            continue
        eligible.append(gene_id)
        names_by_id[gene_id] = name
    if len(eligible) < n_genes:
        raise ValueError(f"only {len(eligible)} shared standard features; need {n_genes}")

    stats: dict[str, dict[str, np.ndarray | int]] = {}
    for condition in conditions:
        path = tables[condition]["path"]
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs = obj.obs
            required = {"guide_group", "guide_type", "low_quality", "total_counts"}
            missing = required.difference(obs.columns)
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            group = np.asarray([_safe_text(v) for v in obs["guide_group"]], dtype=object)
            typ = np.asarray([_safe_text(v).lower() for v in obs["guide_type"]], dtype=object)
            low = np.asarray([_safe_text(v).lower() in {"true", "1", "yes", "t"} for v in obs["low_quality"]], dtype=bool)
            ntc = (group == SINGLE_GUIDE_GROUP) & ~low & np.asarray([bool(re.search(r"ntc|non[-_ ]?target|control|negative", x, re.I)) for x in typ])
            selected = []
            for idx in np.flatnonzero(ntc):
                selected.append(int(idx))
                if len(selected) >= sample_size:
                    break
            indices = [tables[condition]["by_id"][g]["index"] for g in eligible]
            sums = np.zeros(len(indices), dtype=np.float64)
            squares = np.zeros(len(indices), dtype=np.float64)
            used = 0
            # Read contiguous blocks until the deterministic NTC quota is met.
            selected_set = set(selected)
            for start in range(0, len(obs), max(1, int(block_rows))):
                stop = min(len(obs), start + max(1, int(block_rows)))
                local_ntc = np.asarray([i in selected_set for i in range(start, stop)], dtype=bool)
                if not local_ntc.any():
                    continue
                block = _read_csr_block_columns(path, start, stop, indices, obj.n_vars)
                lib = np.asarray(obs["total_counts"].iloc[start:stop], dtype=np.float32)
                lib = np.where(np.isfinite(lib) & (lib > 0), lib, 1.0)
                block = np.log1p(block / lib[:, None] * 1e4)
                values = block[local_ntc]
                sums += values.sum(axis=0)
                squares += np.square(values, dtype=np.float32).sum(axis=0)
                used += int(local_ntc.sum())
                if used >= len(selected):
                    break
            mean = sums / max(used, 1)
            variance = np.maximum(squares / max(used, 1) - np.square(mean), 0.0)
            stats[condition] = {"mean": mean, "variance": variance, "n_ntc": used}
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
    means = np.stack([np.asarray(stats[c]["mean"]) for c in conditions])
    variances = np.stack([np.asarray(stats[c]["variance"]) for c in conditions])
    score = np.mean(variances, axis=0) + np.var(means, axis=0)
    ranked = sorted(range(len(eligible)), key=lambda i: (-float(score[i]), eligible[i]))
    pilot_names = {str(x).upper() for x in pilot_genes if str(x).strip()}
    forced = [i for i, g in enumerate(eligible) if names_by_id[g].upper() in pilot_names or g.upper() in pilot_names]
    selected_indices = list(dict.fromkeys(forced + ranked))[:n_genes]
    selected_ids = [eligible[i] for i in selected_indices]
    records = [{"gene_id": g, "gene_name": names_by_id[g],
                "selection": "forced_pilot" if i in forced else "ntc_hvg"}
               for i, g in zip(selected_indices, selected_ids)]
    payload = {"version": "shared_gene_order.v1", "gene_order": selected_ids,
               "gene_records": records, "shared_universe": len(eligible),
               "conditions": conditions, "feature_indices": {
                   c: [tables[c]["by_id"][g]["index"] for g in selected_ids] for c in conditions},
               "ntc_sample_size_requested": int(sample_size),
               "ntc_sample_size_used": {c: int(stats[c]["n_ntc"]) for c in conditions},
               "algorithm": "D1 NTC log1p(CP10K) mean-variance ranking; forced pilot genes first",
               "pilot_genes_requested": sorted(pilot_names),
               "pilot_genes_found": sorted(names_by_id[g] for g in selected_ids if names_by_id[g].upper() in pilot_names or g.upper() in pilot_names),
               "pilot_genes_unavailable": sorted(pilot_names - {names_by_id[g].upper() for g in eligible} - {g.upper() for g in eligible}),
               "seed": int(seed), "d2_responses_used": False}
    payload["gene_order_hash"] = hashlib.sha256(json.dumps(selected_ids, separators=(",", ":")).encode()).hexdigest()
    payload["artifact_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def build_d1_ntc_latent(paths: Iterable[str | Path], gene_order: Iterable[str],
                        *, sample_per_condition: int = 10000, n_components: int = 50,
                        seed: int = 20260901, block_rows: int = 2048) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a fixed PCA latent space from a bounded D1 NTC sample.

    This is a transparent, reproducible 50-dimensional state representation;
    it is not a response-derived embedding and does not read D2.  A compact
    NTC sample (at most ``sample_per_condition`` cells per D1 condition) is
    gathered by contiguous CSR blocks, then centered PCA is fitted once.
    """
    import anndata as ad
    from sklearn.decomposition import PCA

    files = [Path(p) for p in paths]
    if not files or any("D1" not in p.name for p in files):
        raise ValueError("build_d1_ntc_latent accepts D1 files only")
    genes = list(dict.fromkeys(map(str, gene_order)))
    matrices: list[np.ndarray] = []
    labels: list[str] = []
    used_by_condition: dict[str, int] = {}
    for path in files:
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs = obj.obs
            group = np.asarray([_safe_text(v) for v in obs["guide_group"]], dtype=object)
            typ = np.asarray([_safe_text(v).lower() for v in obs["guide_type"]], dtype=object)
            low = np.asarray([_safe_text(v).lower() in {"true", "1", "yes", "t"} for v in obs["low_quality"]], dtype=bool)
            ntc = (group == SINGLE_GUIDE_GROUP) & ~low & np.asarray([bool(re.search(r"ntc|non[-_ ]?target|control|negative", x, re.I)) for x in typ])
            selected = set(map(int, np.flatnonzero(ntc)[:sample_per_condition]))
            var = obj.var
            ids = [_safe_text(v) for v in var["gene_ids"]] if "gene_ids" in var else [_safe_text(v) for v in obj.var_names]
            names = [_safe_text(v) for v in var["gene_name"]] if "gene_name" in var else [_safe_text(v) for v in obj.var_names]
            lookup = {v: i for i, v in enumerate(ids) if v}
            lookup.update({v: i for i, v in enumerate(names) if v and v not in lookup})
            indices = [lookup[g] for g in genes if g in lookup]
            if len(indices) != len(genes):
                raise ValueError(f"{path.name} cannot align all fixed genes")
            values: list[np.ndarray] = []
            for start in range(0, len(obs), max(1, int(block_rows))):
                stop = min(len(obs), start + max(1, int(block_rows)))
                local = np.asarray([i in selected for i in range(start, stop)], dtype=bool)
                if not local.any():
                    continue
                block = _read_csr_block_columns(path, start, stop, indices, obj.n_vars)
                lib = np.asarray(obs["total_counts"].iloc[start:stop], dtype=np.float32)
                lib = np.where(np.isfinite(lib) & (lib > 0), lib, 1.0)
                values.append(np.log1p(block / lib[:, None] * 1e4)[local].astype(np.float32, copy=False))
                if sum(x.shape[0] for x in values) >= len(selected):
                    break
            matrix = np.vstack(values) if values else np.zeros((0, len(genes)), dtype=np.float32)
            matrices.append(matrix[:len(selected)])
            condition = path.stem.split(".")[0]
            labels.extend([condition] * len(matrices[-1]))
            used_by_condition[condition] = len(matrices[-1])
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
    matrix = np.vstack(matrices)
    if len(matrix) < 2:
        raise ValueError("fewer than two D1 NTC cells available for latent space")
    components = min(int(n_components), matrix.shape[0], matrix.shape[1])
    model = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    latent = model.fit_transform(matrix).astype(np.float32)
    metadata = {"version": "d1_ntc_pca_latent.v1", "dimensions": components,
                "gene_count": len(genes), "conditions": sorted(used_by_condition),
                "ntc_sample_per_condition": int(sample_per_condition),
                "ntc_sample_used": used_by_condition, "algorithm": "centered randomized PCA on D1 NTC log1p(CP10K)",
                "seed": int(seed), "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                "d2_responses_used": False}
    metadata["latent_hash"] = hashlib.sha256(latent.tobytes()).hexdigest()
    return latent, np.asarray(labels, dtype="U32"), metadata


def build_d1_effect_matrix(paths: Iterable[str | Path], genes: Iterable[str],
                            min_cells_per_guide: int = 100, seed: int = 20260901,
                            guide_qc: Mapping | None = None,
                            library_rows: Iterable[Mapping] | None = None,
                            block_rows: int = 8192,
                            primary_genes: Iterable[str] | None = None,
                            challenge_genes: Iterable[str] | None = None,
                            progress_path: str | Path | None = None) -> dict:
    """Build condition-matched D1 guide and robust gene effects.

    The reader deliberately works on HDF5 CSR blocks and categorical integer
    codes.  It never creates a full ``obs`` dataframe or a full expression
    matrix, and it keeps one HDF5 handle open per condition.  The previous
    implementation decoded six 3-million-row columns into Python strings and
    reopened the 150 GB file for every block, which made the real run both
    needlessly memory hungry and extremely slow.

    ``gene_order`` is an immutable list of feature IDs.  ``var.gene_name`` is
    never used as a feature identifier; this prevents a corrected library gene
    name from silently landing on a different column.  If ``guide_qc`` is
    supplied, an observed guide absent from that fixed library report is
    excluded rather than falling back to raw cell labels.
    """
    import h5py

    gene_order = list(dict.fromkeys(map(str, genes)))
    if not gene_order:
        raise ValueError("at least one gene is required")
    files = [Path(p) for p in paths]
    if not files or any("D1" not in p.name for p in files):
        raise ValueError("build_d1_effect_matrix accepts D1 files only")
    if guide_qc is not None and bool(guide_qc.get("d2_responses_used", False)):
        raise ValueError("guide QC indicates D2 responses were used")
    guide_map: dict[str, dict] = {}
    if guide_qc is not None:
        guide_map = {str(x["original_guide_id"]): dict(x)
                     for x in guide_qc.get("guide_rows", [])}
    if library_rows is not None:
        from .guide_correction import _library_index, _correction
        index = _library_index(library_rows)
        for guide in list(guide_map):
            guide_map[guide] = {**guide_map[guide],
                                **_correction(guide, guide_map[guide], index)}

    progress = Path(progress_path) if progress_path is not None else None
    if progress is not None:
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text("", encoding="utf-8")

    def _progress(event: str, **fields) -> None:
        if progress is None:
            return
        row = {"event": event, **fields}
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    _progress("run_started", conditions=[p.name.split(".")[0] for p in files],
              genes=len(gene_order), block_rows=int(block_rows), d2_responses_used=False)

    # condition -> gene -> guide -> effect/count.  Keys are integer indices
    # during the reduction, keeping the hot path free of string operations.
    by_condition: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    guide_counts: dict[str, dict[str, dict[str, int]]] = {}
    condition_meta: dict[str, dict] = {}

    def _codes_for(handle, name: str, start: int, stop: int):
        categories, values = _h5_obs_field(handle, name)
        raw = np.asarray(values[start:stop])
        if categories is None:
            return raw, None
        return raw.astype(np.int64, copy=False), categories

    for path in files:
        # D2 rejection happens before this open.  h5py is used directly so
        # the backed AnnData reader cannot materialise a giant obs dataframe.
        with h5py.File(path, "r") as handle:
            condition = path.name.split(".")[0]
            _progress("condition_started", condition=condition)
            x_node = handle["X"]
            _validate_csr_structure_h5(handle)
            if isinstance(x_node, h5py.Dataset):
                shape = tuple(int(x) for x in x_node.shape)
                n_vars = shape[1]
            else:
                shape_attr = x_node.attrs.get("shape")
                if shape_attr is None:
                    n_vars = int(handle["var"]["_index"].shape[0])
                    n_cells = int(handle["obs"]["_index"].shape[0])
                else:
                    n_cells, n_vars = (int(shape_attr[0]), int(shape_attr[1]))
            n_cells = int(shape[0]) if isinstance(x_node, h5py.Dataset) else int(n_cells)
            obs_node = handle["obs"]
            required = {"guide_id", "guide_type", "guide_group", "low_quality",
                        "total_counts"}
            missing = sorted(name for name in required if name not in obs_node)
            if missing:
                raise ValueError(f"{path.name} missing columns: {missing}")
            var_node = handle["var"]
            if "gene_ids" not in var_node:
                raise ValueError(f"{path.name} has no explicit var.gene_ids")
            gene_ids = _h5_strings(var_node["gene_ids"])
            lookup: dict[str, int] = {}
            for index, gene_id in enumerate(gene_ids):
                if gene_id and gene_id not in lookup:
                    lookup[gene_id] = index
            missing_features = [gene for gene in gene_order if gene not in lookup]
            if missing_features:
                raise ValueError(f"{path.name} missing fixed gene IDs: {missing_features[:5]}"
                                 f" (n={len(missing_features)})")
            col_idx = [lookup[gene] for gene in gene_order]
            gene_pos = {gene: index for index, gene in enumerate(gene_order)}

            guide_categories, guide_dataset = _h5_obs_field(handle, "guide_id")
            group_categories, group_dataset = _h5_obs_field(handle, "guide_group")
            type_categories, type_dataset = _h5_obs_field(handle, "guide_type")
            if guide_categories is None or group_categories is None or type_categories is None:
                raise ValueError(f"{path.name} guide_id/group/type must be categorical")
            group_single = np.asarray([x == SINGLE_GUIDE_GROUP for x in group_categories], dtype=bool)
            type_ntc = np.asarray([bool(re.search(r"ntc|non[-_ ]?target|control|negative", x, re.I))
                                   for x in type_categories], dtype=bool)
            type_target = np.asarray([bool(re.search(r"target", x, re.I)) and not is_ntc
                                      for x, is_ntc in zip(type_categories, type_ntc)], dtype=bool)

            n_guides = len(guide_categories)
            guide_kind = np.zeros(n_guides, dtype=np.int8)  # 1 NTC, 2 target
            guide_gene = np.full(n_guides, -1, dtype=np.int32)
            for code, guide in enumerate(guide_categories):
                info = guide_map.get(guide)
                if guide_qc is not None and info is None:
                    continue  # fixed library authority: unknown is excluded
                if info is None:
                    # Compact fixture path only.  Real runs always pass guide QC.
                    continue
                if (_text(info.get("guide_group")) != SINGLE_GUIDE_GROUP or
                        bool(info.get("low_quality", False)) or
                        not bool(info.get("eligible", False))):
                    continue
                if bool(info.get("is_ntc", False)):
                    guide_kind[code] = 1
                    continue
                if not bool(info.get("is_targeting", False)):
                    continue
                mapped_id = _text(info.get("corrected_target_gene_id"))
                if mapped_id in gene_pos:
                    guide_kind[code] = 2
                    guide_gene[code] = gene_pos[mapped_id]

            # If no fixed guide report is given, use only explicit raw IDs and
            # the same group/type rules.  Raw gene names are intentionally not
            # accepted as feature IDs.
            raw_id_categories, raw_id_dataset = _h5_obs_field(handle, "perturbed_gene_id") \
                if "perturbed_gene_id" in obs_node else (None, None)
            raw_id_to_gene = ({value: gene_pos[value] for value in (raw_id_categories or [])
                              if value in gene_pos} if raw_id_categories is not None else {})

            low_categories = None
            low_dataset = _h5_obs_field(handle, "low_quality")[1]
            total_dataset = _h5_obs_field(handle, "total_counts")[1]
            # ``total_counts`` is required above and is read only per block.
            # It is never replaced by a selected-column sum.
            sums_ntc = np.zeros(len(gene_order), dtype=np.float64)
            n_ntc = 0
            target_sums: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
            target_n: dict[int, Counter] = defaultdict(Counter)
            step = max(1, int(block_rows))
            block_count = 0
            for start in range(0, n_cells, step):
                block_count += 1
                stop = min(n_cells, start + step)
                block = _read_csr_block_columns_h5(handle, start, stop, col_idx, n_vars)
                lib = np.asarray(total_dataset[start:stop], dtype=np.float32)
                lib = np.where(np.isfinite(lib) & (lib > 0), lib, 1.0)
                block = np.log1p(block / lib[:, None] * 1e4).astype(np.float32, copy=False)

                guide_codes = np.asarray(guide_dataset[start:stop], dtype=np.int64)
                group_codes = np.asarray(group_dataset[start:stop], dtype=np.int64)
                type_codes = np.asarray(type_dataset[start:stop], dtype=np.int64)
                low_raw = np.asarray(low_dataset[start:stop])
                low = (low_raw.astype(bool, copy=False) if low_raw.dtype == bool else
                       np.asarray([_text(x).lower() in {"1", "true", "t", "yes", "y"}
                                   for x in low_raw], dtype=bool))
                valid_code = ((guide_codes >= 0) & (guide_codes < n_guides) &
                              (group_codes >= 0) & (group_codes < len(group_single)) &
                              (type_codes >= 0) & (type_codes < len(type_ntc)))
                cell_single = np.zeros(stop - start, dtype=bool)
                cell_ntc_type = np.zeros(stop - start, dtype=bool)
                cell_target_type = np.zeros(stop - start, dtype=bool)
                cell_kind = np.zeros(stop - start, dtype=np.int8)
                cell_gene = np.full(stop - start, -1, dtype=np.int32)
                if valid_code.any():
                    positions = np.flatnonzero(valid_code)
                    cell_single[positions] = group_single[group_codes[positions]]
                    cell_ntc_type[positions] = type_ntc[type_codes[positions]]
                    cell_target_type[positions] = type_target[type_codes[positions]]
                    cell_kind[positions] = guide_kind[guide_codes[positions]]
                    cell_gene[positions] = guide_gene[guide_codes[positions]]
                if guide_qc is None:
                    # Derive target identity from raw IDs only for the small
                    # fixture/compatibility path; production requires QC.
                    if raw_id_categories is not None:
                        raw_codes = np.asarray(raw_id_dataset[start:stop], dtype=np.int64)
                        valid_raw = (raw_codes >= 0) & (raw_codes < len(raw_id_categories))
                        raw_gene = np.asarray([raw_id_to_gene.get(raw_id_categories[c], -1)
                                               if valid_raw[i] else -1
                                               for i, c in enumerate(raw_codes)], dtype=np.int32)
                        cell_gene = np.where(raw_gene >= 0, raw_gene, cell_gene)
                        cell_kind = np.where((cell_gene >= 0) & cell_target_type & cell_single, 2, cell_kind)
                        cell_kind = np.where(cell_ntc_type & cell_single, 1, cell_kind)
                ntc_mask = (~low) & cell_single & cell_ntc_type & (cell_kind == 1)
                target_mask = (~low) & cell_single & cell_target_type & (cell_kind == 2) & (cell_gene >= 0)
                if ntc_mask.any():
                    sums_ntc += block[ntc_mask].sum(axis=0, dtype=np.float64)
                    n_ntc += int(ntc_mask.sum())
                if target_mask.any():
                    target_codes = guide_codes[target_mask]
                    for code in np.unique(target_codes):
                        gene_index = int(guide_gene[code])
                        if gene_index < 0:
                            continue
                        code_mask = target_mask & (guide_codes == code)
                        target_sums[gene_index][int(code)] = (
                            target_sums[gene_index].get(int(code), np.zeros(len(gene_order), dtype=np.float64)) +
                            block[code_mask].sum(axis=0, dtype=np.float64))
                        target_n[gene_index][int(code)] += int(code_mask.sum())
                if block_count == 1 or block_count % 64 == 0 or stop == n_cells:
                    _progress("block_completed", condition=condition, block=block_count,
                              cells_processed=stop, total_cells=n_cells, ntc_cells=n_ntc)

            ntc_mean = sums_ntc / n_ntc if n_ntc else np.zeros(len(gene_order), dtype=np.float64)
            condition_out: dict[str, dict[str, np.ndarray]] = {}
            count_out: dict[str, dict[str, int]] = defaultdict(dict)
            for gene_index, guide_values in target_sums.items():
                gene = gene_order[gene_index]
                for code, summed in guide_values.items():
                    n = int(target_n[gene_index][code])
                    if n < min_cells_per_guide:
                        continue
                    guide = guide_categories[code]
                    condition_out.setdefault(gene, {})[guide] = summed / n - ntc_mean
                    count_out[gene][guide] = n
            by_condition[condition] = condition_out
            guide_counts[condition] = count_out
            condition_meta[condition] = {
                "n_cells": n_cells, "ntc_cells": n_ntc,
                "target_cells": int(sum(count for values in count_out.values() for count in values.values())),
                "library_size_field": "total_counts", "missing_features": [],
                "block_rows": int(step), "reader": "h5py_csr_block_categorical_codes",
            }
            _progress("condition_completed", condition=condition, blocks=block_count,
                      ntc_cells=n_ntc, guides=sum(len(x) for x in count_out.values()))

    conditions = sorted(by_condition)
    guide_rows: list[dict] = []
    gene_effects: dict[str, dict[str, list[float]]] = {}
    gene_qc: list[dict] = []
    for condition in conditions:
        gene_effects[condition] = {}
        for gene in gene_order:
            effects = by_condition[condition].get(gene, {})
            values = list(effects.values())
            if values:
                stacked = np.stack(values)
                median = np.median(stacked, axis=0)
                gene_effects[condition][gene] = median.tolist()
                signs = np.sign(stacked)
                consistency = float(np.mean(np.abs(np.mean(signs, axis=0))))
                conflict = consistency < 0.5
                gene_qc.append({"condition": condition, "gene": gene,
                                "n_guides": len(values), "guide_consistency": consistency,
                                "effect_l2": float(np.linalg.norm(median)),
                                "unstable": conflict, "conflict": conflict,
                                "conflict_reason": "discordant_guides" if conflict else ""})
            else:
                gene_effects[condition][gene] = [0.0] * len(gene_order)
                gene_qc.append({"condition": condition, "gene": gene,
                                "n_guides": 0, "guide_consistency": 0.0,
                                "effect_l2": 0.0, "unstable": True, "conflict": True,
                                "conflict_reason": "no_eligible_guide"})
            for guide, effect in sorted(effects.items()):
                guide_rows.append({"condition": condition, "gene": gene, "guide_id": guide,
                                   "n_cells": int(guide_counts[condition][gene][guide]),
                                   "effect": effect.tolist(), "normalization": "log1p(CP10K)",
                                   "library_size_field": "total_counts"})
    for gene in gene_order:
        cond_values = [np.asarray(gene_effects[c][gene], dtype=np.float32) for c in conditions]
        stacked = np.stack(cond_values)
        norms = np.linalg.norm(stacked, axis=1)
        gene_qc.append({"condition": "across_conditions", "gene": gene,
                        "n_conditions": len(cond_values),
                        "stimulation_dependence_l2_range": float(np.max(norms) - np.min(norms)),
                        "stimulation_dependence_cv": float(np.std(norms) / max(np.mean(norms), 1e-12)),
                        "unstable": bool(np.max(norms) - np.min(norms) > max(np.mean(norms), 1e-12)),
                        "conflict": False, "conflict_reason": ""})

    def _pilot_qc(selected: Iterable[str] | None) -> dict:
        selected_set = {str(x) for x in (selected or ())}
        rows = [row for row in gene_qc if row["condition"] in conditions and row["gene"] in selected_set]
        return {"n_genes": len(selected_set), "n_rows": len(rows),
                "genes_with_effect": sum(row["n_guides"] > 0 for row in rows),
                "unstable_rows": sum(bool(row["unstable"]) for row in rows),
                "rows": rows}

    payload = {
        "version": "d1_effect_matrix.v2", "gene_order": gene_order,
        "gene_order_hash": hashlib.sha256(json.dumps(gene_order, separators=(",", ":")).encode()).hexdigest(),
        "guide_effects": guide_rows, "gene_effects": gene_effects, "gene_qc": gene_qc,
        "pilot_qc": {"primary_64": _pilot_qc(primary_genes),
                     "challenge_32": _pilot_qc(challenge_genes)},
        "condition_metadata": condition_meta, "conditions": conditions,
        "min_cells_per_guide": int(min_cells_per_guide), "seed": int(seed),
        "normalization": "log1p(CP10K)",
        "guide_filter": {"guide_group": SINGLE_GUIDE_GROUP,
                         "ntc_guide_type": "non-targeting/NTC",
                         "target_guide_type": "targeting", "library_mapping_required": guide_qc is not None},
        "d2_responses_used": False,
    }
    payload["effect_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _progress("run_completed", guide_rows=len(guide_rows), effect_hash=payload["effect_hash"],
              d2_responses_used=False)
    return payload
