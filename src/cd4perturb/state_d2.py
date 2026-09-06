from __future__ import annotations

"""D2-specific, auditable preparation utilities for a single-step STATE run.

The module intentionally contains no model code.  It freezes the objects that
must be identical for Scratch and Transfer (program definitions, metadata
contract, perturbation vocabulary, split manifest and gene panel).  Large
AnnData files are inspected in backed mode and expression is read in bounded
CSR blocks; callers never need to densify a complete experiment.
"""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .guide_correction import SINGLE_GUIDE_GROUP, _text, load_guide_library


IDENTITY_PROGRAMS = {
    "Naive": ["TCF7", "LEF1", "CCR7", "SELL", "MAL"],
    "Th1": ["TBX21", "IL12RB2", "CXCR3"],
    "Th2": ["GATA3", "CCR4", "PTGDR2"],
    "Th17": ["RORC", "CCR6", "IL23R"],
}
EFFECT_GENES = {
    "Th1": ["IFNG"],
    "Th2": ["IL4", "IL5", "IL13"],
    "Th17": ["IL17A", "IL17F", "CCL20", "IL26"],
}
CONFOUNDER_PROGRAMS = {
    "activation": ["FOS", "JUN", "DUSP1", "CD69", "IL2RA", "TNFRSF4"],
    "stress": ["HSPA1A", "HSPA1B", "DNAJB1", "HSP90AA1", "XBP1"],
    "apoptosis": ["BAX", "BBC3", "PMAIP1", "CASP3", "FAS"],
    "cell_cycle": ["MKI67", "TOP2A", "TYMS", "PCNA", "STMN1"],
    "interferon": ["ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "IFI6"],
}
REQUIRED_CONDITIONS = ("Rest", "Stim8hr", "Stim48hr")


def _hash_payload(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _bool_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "t", "yes", "y"}


def _obs_equals(series, value: str) -> np.ndarray:
    if hasattr(series.dtype, "categories"):
        categories = np.asarray([_text(x) for x in series.cat.categories])
        hits = np.flatnonzero(categories == str(value))
        return np.isin(series.cat.codes.to_numpy(), hits)
    return np.asarray([_text(value_) == str(value) for value_ in series.to_numpy()])


def _obs_category_values(series) -> tuple[np.ndarray, np.ndarray] | tuple[None, np.ndarray]:
    if hasattr(series.dtype, "categories"):
        return (np.asarray([_text(x) for x in series.cat.categories]),
                series.cat.codes.to_numpy())
    return None, np.asarray([_text(x) for x in series.to_numpy()])


def _condition(path: str | Path) -> str:
    match = re.search(r"_(Rest|Stim8hr|Stim48hr)(?:\.|$)", Path(path).name)
    if not match:
        raise ValueError(f"cannot infer D2 condition from filename: {path}")
    return match.group(1)


def _require_d2_paths(paths: Iterable[str | Path]) -> list[Path]:
    result = [Path(path) for path in paths]
    if len(result) != 3 or any(not path.name.startswith("D2_") for path in result):
        raise ValueError("D2 operations require exactly three D2 files")
    if {_condition(path) for path in result} != set(REQUIRED_CONDITIONS):
        raise ValueError("D2 files must contain Rest, Stim8hr and Stim48hr")
    return sorted(result, key=lambda path: REQUIRED_CONDITIONS.index(_condition(path)))


def load_lineage_programs(path: str | Path = "config/th_lineage_programs.v1.json") -> dict:
    """Load and validate the frozen program definition without expression data."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("identity_programs", "effect_genes", "confounder_programs", "anchor_policy"):
        if key not in payload:
            raise ValueError(f"lineage program config missing {key}")
    validate_lineage_programs(payload)
    return payload


def validate_lineage_programs(payload: Mapping) -> bool:
    identity = payload["identity_programs"]
    if set(identity) != {"Naive", "Th1", "Th2", "Th17"}:
        raise ValueError("identity programs must be exactly Naive/Th1/Th2/Th17")
    seen: set[str] = set()
    for program, genes in identity.items():
        if not genes or len(set(genes)) != len(genes):
            raise ValueError(f"identity program {program} is empty or duplicated")
        overlap = seen.intersection(genes)
        if overlap:
            raise ValueError(f"identity programs overlap: {sorted(overlap)}")
        seen.update(genes)
    for key in ("effect_genes", "confounder_programs"):
        for name, genes in payload[key].items():
            if not genes or len(set(genes)) != len(genes):
                raise ValueError(f"{key}/{name} is empty or duplicated")
    policy = payload["anchor_policy"]
    if int(policy.get("max_forced_genes", 0)) > 10:
        raise ValueError("at most ten identity anchors may be forced")
    if tuple(policy.get("required_conditions", ())) != REQUIRED_CONDITIONS:
        raise ValueError("required D2 conditions are not frozen")
    return True


def _rank_auc(values: np.ndarray, positive: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    if positive.sum() == 0 or positive.sum() == len(values):
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def score_programs(expression: np.ndarray, gene_names: Sequence[str], programs: Mapping | None = None) -> dict[str, np.ndarray]:
    """Score programs as the mean of per-gene cell-wise z-scores.

    Missing genes are omitted from a score and reported through the companion
    ``program_coverage`` helper.  This prevents a missing cytokine from being
    silently interpreted as a negative lineage score.
    """
    matrix = np.asarray(expression, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(gene_names):
        raise ValueError("expression must be cells × genes and match gene_names")
    definitions = programs or {"identity_programs": IDENTITY_PROGRAMS,
                               "effect_genes": EFFECT_GENES,
                               "confounder_programs": CONFOUNDER_PROGRAMS}
    lookup = {str(g): i for i, g in enumerate(gene_names)}
    result: dict[str, np.ndarray] = {}
    for group in ("identity_programs", "effect_genes", "confounder_programs"):
        for name, genes in definitions.get(group, {}).items():
            columns = [lookup[g] for g in genes if g in lookup]
            if not columns:
                result[name] = np.full(matrix.shape[0], np.nan)
                continue
            subset = matrix[:, columns]
            mean = np.nanmean(subset, axis=0)
            scale = np.nanstd(subset, axis=0)
            scale = np.where(scale > 1e-8, scale, 1.0)
            prefix = {"effect_genes": "effect", "confounder_programs": "confounder"}.get(group, group)
            key = name if group == "identity_programs" else f"{prefix}:{name}"
            result[key] = np.nanmean((subset - mean) / scale, axis=1)
    return result


def program_coverage(gene_names: Sequence[str], programs: Mapping | None = None) -> dict:
    definitions = programs or {"identity_programs": IDENTITY_PROGRAMS,
                               "effect_genes": EFFECT_GENES,
                               "confounder_programs": CONFOUNDER_PROGRAMS}
    measured = set(map(str, gene_names))
    out = {}
    for group in ("identity_programs", "effect_genes", "confounder_programs"):
        out[group] = {name: {"measured": [g for g in genes if g in measured],
                             "missing": [g for g in genes if g not in measured],
                             "fraction": sum(g in measured for g in genes) / len(genes)}
                      for name, genes in definitions.get(group, {}).items()}
    return out


def validate_program_scores(scores: Mapping[str, np.ndarray], labels: Sequence[str], donors: Sequence[str]) -> dict:
    """Validate external labels without requiring scikit-learn.

    A donor contributes a direction only when both positive and negative cells
    are present.  The result keeps per-donor AUCs so the 2/3 consistency rule
    cannot be hidden by pooling cells from one donor.
    """
    labels = np.asarray(labels, dtype=str)
    donors = np.asarray(donors, dtype=str)
    if len(labels) != len(donors) or any(len(v) != len(labels) for v in scores.values()):
        raise ValueError("program scores, labels and donors must have equal length")
    per_program = {}
    for program, values in scores.items():
        rows = {}
        for donor in sorted(set(donors)):
            mask = donors == donor
            auc = _rank_auc(np.asarray(values)[mask], labels[mask] == program)
            rows[donor] = auc
        valid = [value for value in rows.values() if value is not None]
        direction = sum(value >= 0.5 for value in valid)
        per_program[program] = {"donor_auc": rows,
                                "direction_consistent": bool(valid) and direction >= math.ceil(2 * len(valid) / 3),
                                "macro_auc": float(np.mean(valid)) if valid else None}
    macro = [v["macro_auc"] for v in per_program.values() if v["macro_auc"] is not None]
    return {"programs": per_program, "macro_auc": float(np.mean(macro)) if macro else None,
            "minimum_donor_direction_consistency": "2/3"}


def _var_gene_columns(var) -> tuple[list[str], list[str]]:
    ids = [str(x) for x in var.index]
    for column in ("gene_id", "ensembl_id", "gene_ids", "feature_id"):
        if column in var.columns:
            ids = [_text(x) for x in var[column].tolist()]
            break
    symbols = ids[:]
    for column in ("gene_name", "gene_symbol", "symbol", "gene_names"):
        if column in var.columns:
            symbols = [_text(x) for x in var[column].tolist()]
            break
    return ids, symbols


def audit_d2_data(paths: Iterable[str | Path], csr_summary: Mapping | None = None) -> dict:
    """Create a metadata-only D2 contract and verify the common gene axis."""
    paths = _require_d2_paths(paths)
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - exercised on the remote env
        raise RuntimeError("D2 metadata audit requires anndata") from exc
    files = []
    axes = []
    for path in paths:
        obj = ad.read_h5ad(path, backed="r")
        try:
            if obj.n_vars == 0 or obj.n_obs == 0:
                raise ValueError(f"empty AnnData object: {path}")
            ids, symbols = _var_gene_columns(obj.var)
            if len(set(ids)) != len(ids) or len(set(symbols)) != len(symbols):
                raise ValueError(f"D2 gene axis is not unique: {path}")
            axes.append(ids)
            sample = obj.X[: min(64, obj.n_obs), : min(256, obj.n_vars)]
            sample = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
            files.append({"path": str(path), "file_name": path.name,
                          "condition": _condition(path), "n_obs": int(obj.n_obs),
                          "n_vars": int(obj.n_vars),
                          "var_id_hash": _hash_payload(ids),
                          "gene_symbol_hash": _hash_payload(symbols),
                          "gene_id_column": next((x for x in ("gene_id", "ensembl_id", "gene_ids", "feature_id") if x in obj.var.columns), "_index"),
                          "gene_symbol_column": next((x for x in ("gene_name", "gene_symbol", "symbol", "gene_names") if x in obj.var.columns), "_index"),
                          "obs_columns": sorted(map(str, obj.obs.columns)),
                          "layers": sorted(map(str, obj.layers.keys())),
                          "has_raw": obj.raw is not None,
                          "x_storage": type(obj.X).__name__,
                          "sample_nonnegative": bool(np.all(sample >= 0)),
                          "sample_integer_like": bool(np.allclose(sample, np.round(sample))),
                          "sample_finite": bool(np.isfinite(sample).all()),
                          "puro_r_present": "PuroR" in symbols})
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
    if any(set(axis) != set(axes[0]) for axis in axes[1:]):
        raise ValueError("D2 files do not share the same gene ID set")
    canonical_ids = sorted(axes[0])
    order_equal = all(axis == axes[0] for axis in axes[1:])
    required = {"guide_id", "guide_type", "guide_group", "low_quality",
                "perturbed_gene_name", "perturbed_gene_id"}
    missing = sorted(required.difference(files[0]["obs_columns"]))
    if missing:
        raise ValueError(f"D2 obs contract missing columns: {missing}")
    return {"version": "d2_data_audit.v1", "donor_id": "D2",
            "conditions": list(REQUIRED_CONDITIONS), "files": files,
            "common_n_vars": len(canonical_ids), "common_gene_axis_hash": _hash_payload(canonical_ids),
            "gene_axis_order_equal": order_equal,
            "canonical_gene_axis": "sorted Ensembl gene IDs; per-file column maps are applied at read time",
            "raw_count_semantics": "sampled_nonnegative_integer_like; full CSR validity is supplied by the external audit",
            "required_obs_columns": sorted(required),
            "csr_audit_summary": dict(csr_summary or {}),
            "d2_responses_used": False,
            "raw_h5ad_modified": False}


def build_d2_perturbation_vocab(paths: Iterable[str | Path], library_rows: Iterable[Mapping] | None = None) -> dict:
    """Freeze a name-based perturbation vocabulary; NTC is always index zero."""
    paths = _require_d2_paths(paths)
    if library_rows is not None:
        from .guide_correction import audit_guides
        report = audit_guides(paths, list(library_rows), donor_id="D2")
        rows = report["guide_rows"]
    else:
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("D2 vocabulary requires anndata") from exc
        rows = []
        seen = set()
        for path in paths:
            obj = ad.read_h5ad(path, backed="r")
            try:
                columns = ["guide_id", "guide_type", "guide_group", "low_quality",
                           "perturbed_gene_name", "perturbed_gene_id"]
                for values in obj.obs[columns].itertuples(index=False, name=None):
                    guide, guide_type, group, low, raw_name, raw_id = values
                    key = _text(guide)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    is_ntc = bool(re.search(r"ntc|non[-_ ]?target|control|negative", _text(guide_type), re.I))
                    target = _text(raw_name) or _text(raw_id)
                    rows.append({"original_guide_id": key, "corrected_target_gene_name": "" if is_ntc else target,
                                 "corrected_target_gene_id": "", "is_ntc": is_ntc,
                                 "is_targeting": not is_ntc, "eligible": group == SINGLE_GUIDE_GROUP and not _bool_value(low)})
            finally:
                if getattr(obj, "file", None) is not None:
                    obj.file.close()
    targets = sorted({(_text(row.get("corrected_target_gene_name")) or _text(row.get("corrected_target_gene_id")))
                      for row in rows if row.get("is_targeting") and row.get("eligible")})
    vocab = ["NTC"] + [target for target in targets if target and target != "NTC"]
    target_to_guides = {target: sorted({row["original_guide_id"] for row in rows
                                        if row.get("is_targeting") and row.get("eligible") and
                                        ((_text(row.get("corrected_target_gene_name")) or _text(row.get("corrected_target_gene_id"))) == target)})
                        for target in vocab[1:]}
    return {"version": "d2_perturbation_vocab.v1", "donor_id": "D2",
            "index_rule": "NTC=0; remaining entries sorted by corrected target name",
            "perturbation_name_to_integer": {name: i for i, name in enumerate(vocab)},
            "perturbation_names": vocab, "target_to_guides": target_to_guides,
            "unique_observed_guides": len(rows),
            "eligible_target_count": len(targets), "d2_responses_used": False,
            "vocab_hash": _hash_payload(vocab)}


def freeze_d2_splits(target_genes: Iterable[str], seed: int = 20260901,
                     conditions: Sequence[str] = REQUIRED_CONDITIONS) -> dict:
    """Freeze gene×background records with disjoint train/validation/test combos.

    Held-out genes retain their other backgrounds in training; only the target
    condition response is held out.  Validation and test genes are disjoint,
    while NTC cells remain available for the background distribution.
    """
    genes = sorted({str(g) for g in target_genes if str(g) and str(g) != "NTC"})
    if not genes:
        raise ValueError("at least one target gene is required")
    conditions = tuple(conditions)
    if tuple(conditions) != REQUIRED_CONDITIONS:
        raise ValueError("D2 split conditions are frozen to Rest/Stim8hr/Stim48hr")
    rng = np.random.default_rng(int(seed))
    order = list(genes)
    rng.shuffle(order)
    n_test = max(1, round(len(order) * 0.2))
    n_validation = max(1, round(len(order) * 0.2)) if len(order) > 2 else 0
    test_genes = set(order[:n_test])
    validation_genes = set(order[n_test:n_test + n_validation])
    records = []
    for gene in genes:
        heldout = "test" if gene in test_genes else "validation" if gene in validation_genes else None
        heldout_condition = conditions[hashlib.sha256(f"{seed}:{gene}".encode()).digest()[0] % len(conditions)] if heldout else None
        for condition in conditions:
            split = heldout if heldout and condition == heldout_condition else "train"
            records.append({"perturbation_name": gene, "condition": condition, "split": split,
                            "response_used_for_training": split == "train"})
    # The same NTC background is explicitly registered, but never called a
    # real pre-perturbation state.
    ntc = [{"perturbation_name": "NTC", "condition": condition, "split": "train",
            "response_used_for_training": True} for condition in conditions]
    records.extend(ntc)
    return {"version": "d2_splits.v1", "seed": int(seed),
            "scheme": "gene_x_biological_background_leaveout",
            "records": records, "test_genes": sorted(test_genes),
            "validation_genes": sorted(validation_genes),
            "non_targeting_background": "NTC train distribution; population state region only",
            "split_hash": _hash_payload(records), "d2_responses_used": False}


def apply_identity_anchor_policy(raw_hvg: Sequence[str], gene_stats: Mapping[str, Mapping],
                                 programs: Mapping | None = None, max_forced_genes: int = 10) -> dict:
    """Apply the pre-registered, at-most-ten identity-anchor substitutions."""
    if len(raw_hvg) != 2000 or len(set(raw_hvg)) != 2000:
        raise ValueError("raw_hvg must contain exactly 2,000 unique genes")
    definitions = (programs or {"identity_programs": IDENTITY_PROGRAMS})["identity_programs"]
    anchors = [gene for genes in definitions.values() for gene in genes]
    eligible = []
    for position, gene in enumerate(anchors):
        stats = gene_stats.get(gene, {})
        detected = stats.get("detected_by_condition", {})
        train_detected = int(sum(bool(detected.get(condition, 0)) for condition in REQUIRED_CONDITIONS[:2]))
        detected_all = all(int(detected.get(condition, 0)) > 0 for condition in REQUIRED_CONDITIONS)
        if (gene not in raw_hvg and (stats.get("measured_in_all_conditions", False) or detected_all) and
                detected_all and
                train_detected >= 2 and int(stats.get("total_detected_cells", 0)) >= 500 and
                int(stats.get("raw_hvg_rank", 10**9)) > 5000):
            eligible.append((position, gene))
    eligible = eligible[: int(max_forced_genes)]
    panel = list(raw_hvg)
    substitutions = []
    for _, gene in eligible:
        candidates = [i for i, item in enumerate(panel)
                      if item not in anchors and item not in {x[1] for x in substitutions}]
        if not candidates:
            break
        index = candidates[-1]
        replaced = panel[index]
        panel[index] = gene
        substitutions.append((replaced, gene))
    if len(panel) != 2000 or len(set(panel)) != 2000:
        raise AssertionError("identity anchor substitution changed panel cardinality")
    return {"version": "d2_gene_panel_2000.v1", "gene_order": panel,
            "gene_order_hash": _hash_payload(panel), "raw_hvg": list(raw_hvg),
            "raw_hvg_hash": _hash_payload(list(raw_hvg)),
            "forced_identity_anchors": [new for _, new in substitutions],
            "substitutions": [{"replaced": old, "forced": new} for old, new in substitutions],
            "forced_count": len(substitutions), "max_forced_genes": int(max_forced_genes),
            "policy": "only measured, unique, two training-condition detected, >=500 cells, raw rank >5000; one-for-one tail replacement"}


def validate_d2_freeze_artifacts(audit: Mapping, vocab: Mapping, splits: Mapping,
                                 panel: Mapping, pilot: Mapping | None = None) -> dict:
    """Cross-check that every downstream consumer sees one frozen contract."""
    if audit.get("donor_id") != "D2" or audit.get("raw_h5ad_modified") is not False:
        raise ValueError("D2 audit is missing or indicates modified raw input")
    names = list(vocab.get("perturbation_names", []))
    if not names or names[0] != "NTC" or vocab.get("perturbation_name_to_integer", {}).get("NTC") != 0:
        raise ValueError("D2 vocabulary must reserve NTC at integer index zero")
    genes = list(panel.get("gene_order", []))
    if len(genes) != 2000 or len(set(genes)) != 2000:
        raise ValueError("D2 panel must contain 2,000 unique genes")
    records = list(splits.get("records", []))
    combos = [(str(row.get("perturbation_name")), str(row.get("condition"))) for row in records]
    if len(combos) != len(set(combos)):
        raise ValueError("D2 split records contain overlapping gene×condition combinations")
    result = {"version": "d2_freeze_validation.v1", "audit_version": audit.get("version"),
              "gene_order_hash": panel.get("gene_order_hash"),
              "perturbation_vocab_hash": vocab.get("vocab_hash"),
              "split_hash": splits.get("split_hash"), "n_genes": len(genes),
              "n_perturbations": len(names), "n_split_records": len(records),
              "pilot_checked": pilot is not None, "d2_responses_used": False}
    if pilot is not None:
        if pilot.get("output_shape") != [32, 2000] or pilot.get("finite") is not True or pilot.get("checkpoint_reload") is not True:
            raise ValueError("D2 pilot contract is not complete")
        if pilot.get("gene_order_hash") != result["gene_order_hash"] or pilot.get("perturbation_vocab_hash") != result["perturbation_vocab_hash"]:
            raise ValueError("D2 pilot used a different panel or vocabulary")
    result["freeze_hash"] = _hash_payload(result)
    return result


def compute_d2_hvg_panel(paths: Iterable[str | Path], output: str | Path | None = None,
                         block_rows: int = 1024, max_cells: int | None = None,
                         seed: int = 20260901, splits: Mapping | None = None) -> dict:
    """Compute a deterministic streaming Seurat-compatible HVG ranking.

    The implementation follows Scanpy's Seurat dispersion convention
    (CP10K→log1p, mean bins, normalized dispersion) while reading raw CSR in
    bounded blocks.  It deliberately exposes the algorithm and row counts in
    the artifact so a later full Scanpy cross-check can be audited rather than
    silently substituted.
    """
    paths = _require_d2_paths(paths)
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    import h5py
    from .data import _read_csr_block_columns_h5
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("D2 HVG computation requires anndata") from exc
    rng = np.random.default_rng(int(seed))
    heldout_pairs = {(str(row["perturbation_name"]), str(row["condition"]))
                     for row in (splits or {}).get("records", [])
                     if row.get("split") != "train" and row.get("perturbation_name") != "NTC"}
    # Metadata pass: common symbols and quality masks, with deterministic
    # reservoir sampling when max_cells is used for a pilot/pre-audit.
    selected: dict[Path, np.ndarray] = {}
    axis_ids: list[str] | None = None
    columns_by_path: dict[Path, list[int]] = {}
    active_columns_by_path: dict[Path, list[int]] = {}
    original_n_vars: dict[Path, int] = {}
    symbols_by_id: dict[str, str] = {}
    for path in paths:
        obj = ad.read_h5ad(path, backed="r")
        try:
            current_ids, current_symbols = _var_gene_columns(obj.var)
            if axis_ids is None:
                axis_ids = sorted(current_ids)
                symbols_by_id = dict(zip(current_ids, current_symbols))
            elif set(current_ids) != set(axis_ids):
                raise ValueError("D2 HVG files have different gene ID sets")
            current_index = {gene_id: index for index, gene_id in enumerate(current_ids)}
            columns_by_path[path] = [current_index[gene_id] for gene_id in axis_ids]
            original_n_vars[path] = len(current_ids)
            required = {"guide_group", "low_quality"}
            if heldout_pairs:
                required.update({"perturbed_gene_name", "perturbed_gene_id"})
            if not required.issubset(obj.obs.columns):
                raise ValueError("D2 HVG requires guide_group and low_quality columns")
            mask = _obs_equals(obj.obs["guide_group"], SINGLE_GUIDE_GROUP)
            quality_values = obj.obs["low_quality"].to_numpy()
            quality = quality_values if quality_values.dtype == bool else np.asarray([_bool_value(value) for value in quality_values])
            mask &= ~quality
            if heldout_pairs:
                condition = _condition(path)
                name_categories, name_codes = _obs_category_values(obj.obs["perturbed_gene_name"])
                id_categories, id_codes = _obs_category_values(obj.obs["perturbed_gene_id"])
                heldout = np.zeros(len(mask), dtype=bool)
                if name_categories is not None:
                    name_hits = {i for i, value in enumerate(name_categories) if (value, condition) in heldout_pairs}
                    heldout |= np.isin(name_codes, list(name_hits))
                else:
                    heldout |= np.asarray([(value, condition) in heldout_pairs for value in name_codes])
                if id_categories is not None:
                    id_hits = {i for i, value in enumerate(id_categories) if (value, condition) in heldout_pairs}
                    heldout |= np.isin(id_codes, list(id_hits))
                else:
                    heldout |= np.asarray([(value, condition) in heldout_pairs for value in id_codes])
                # NTC rows have no held-out pair and remain in the background.
                mask &= ~heldout
            indices = np.flatnonzero(mask)
            if max_cells is not None and len(indices) > max_cells:
                indices = np.sort(rng.choice(indices, size=max_cells, replace=False))
            selected[path] = indices.astype(np.int64)
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
    assert axis_ids is not None
    symbols = [symbols_by_id[gene_id] for gene_id in axis_ids]
    active_positions = [i for i, symbol in enumerate(symbols) if symbol != "PuroR"]
    active_ids = [axis_ids[i] for i in active_positions]
    symbols = [symbols[i] for i in active_positions]
    n_vars = len(symbols)
    for path in paths:
        active_columns_by_path[path] = [columns_by_path[path][i] for i in active_positions]
    sums = np.zeros(n_vars, dtype=np.float64)
    squares = np.zeros(n_vars, dtype=np.float64)
    detected = np.zeros(n_vars, dtype=np.int64)
    detected_by_condition = {condition: np.zeros(n_vars, dtype=np.int64)
                             for condition in REQUIRED_CONDITIONS}
    n_cells = 0
    for path in paths:
        rows = selected[path]
        with h5py.File(path, "r") as handle:
            n_obs = int(handle["obs"]["_index"].shape[0])
            if max_cells is not None:
                # A capped pre-audit must not scan every row of a multi-million
                # cell file.  Group nearby sampled rows into bounded windows.
                windows = []
                cursor = 0
                while cursor < len(rows):
                    end = cursor + 1
                    while (end < len(rows) and end - cursor < block_rows and
                           int(rows[end] - rows[end - 1]) <= 64):
                        end += 1
                    window_start = int(rows[cursor])
                    window_stop = int(rows[end - 1]) + 1
                    local_rows = (rows[cursor:end] - window_start).astype(np.int64, copy=False)
                    windows.append((window_start, window_stop, local_rows))
                    cursor = end
            else:
                windows = []
                for window_start in range(0, n_obs, block_rows):
                    window_stop = min(window_start + block_rows, n_obs)
                    left = int(np.searchsorted(rows, window_start, side="left"))
                    right = int(np.searchsorted(rows, window_stop, side="left"))
                    local_rows = (rows[left:right] - window_start).astype(np.int64, copy=False)
                    if len(local_rows):
                        windows.append((window_start, window_stop, local_rows))
            for window_start, window_stop, local_rows in windows:
                # Read contiguous CSR windows once, then retain only the
                # quality-passing rows.  This is the critical memory-bound
                # path: at most ``block_rows × n_vars`` is materialised.
                full = _read_csr_block_columns_h5(handle, window_start, window_stop,
                                                  active_columns_by_path[path],
                                                  original_n_vars[path])
                block = full[local_rows]
                totals = block.sum(axis=1)
                totals = np.where(totals > 0, totals, 1.0)
                values = np.log1p(block / totals[:, None] * 10000.0).astype(np.float64)
                sums += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                detected += np.count_nonzero(values, axis=0)
                detected_by_condition[_condition(path)] += np.count_nonzero(values, axis=0)
                n_cells += len(values)
    if n_cells < 2:
        raise ValueError("not enough eligible D2 cells for HVG computation")
    means = sums / n_cells
    variances = np.maximum(squares / n_cells - means * means, 0.0)
    dispersion = variances / np.maximum(means, 1e-12)
    # Scanpy's ``flavor=seurat`` bins by mean and normalizes dispersion inside
    # each bin.  Equal-frequency bins make ties deterministic for sparse data.
    order = np.argsort(means, kind="mergesort")
    normalized = np.full(n_vars, -np.inf, dtype=float)
    for group in np.array_split(order, min(20, n_vars)):
        if len(group) == 0:
            continue
        center = float(np.mean(dispersion[group]))
        scale = float(np.std(dispersion[group]))
        normalized[group] = (dispersion[group] - center) / (scale if scale > 1e-12 else 1.0)
    ranking = sorted(range(n_vars), key=lambda i: (-normalized[i], -dispersion[i], symbols[i]))
    raw_hvg = [symbols[i] for i in ranking[:2000]]
    stats = {symbols[i]: {"mean": float(means[i]), "variance": float(variances[i]),
                          "dispersion": float(dispersion[i]), "normalized_dispersion": float(normalized[i]),
                          "raw_hvg_rank": int(ranking.index(i) + 1),
                          "total_detected_cells": int(detected[i]),
                          "detected_by_condition": {condition: int(detected_by_condition[condition][i])
                                                    for condition in REQUIRED_CONDITIONS},
                          "measured_in_all_conditions": True} for i in range(n_vars)}
    result = {"version": "d2_gene_panel_2000.v1" if max_cells is None else "d2_hvg_precheck.v1",
              "method": "streaming_scanpy_seurat_compatible",
              "scanpy_parameters": {"target_sum": 10000, "log1p": True, "flavor": "seurat", "n_bins": 20},
              "block_rows": int(block_rows), "max_cells_per_condition": max_cells,
              "n_cells": int(n_cells), "n_vars": int(n_vars), "conditions": list(REQUIRED_CONDITIONS),
              "raw_hvg": raw_hvg, "gene_statistics": stats,
              "raw_hvg_hash": _hash_payload(raw_hvg),
              "split_hash": (splits or {}).get("split_hash"),
              "heldout_response_pairs_excluded": len(heldout_pairs),
              "d2_responses_used": False}
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
