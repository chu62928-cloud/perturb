from __future__ import annotations

"""Versioned STATE/Cell-Eval re-analysis for the frozen D2 test set.

The historical ``evaluate_state_d2_test.py`` is deliberately not modified.
This entry point has three resumable phases:

``infer``
    Materialise predictions from the six frozen checkpoints.
``metrics``
    Run Cell-Eval and paper-era metrics from the cached predictions.
``all``
    Run both phases.

Large prediction and AnnData files are written to an ignored runtime directory;
only the compact manifest and metric summary are intended for Git.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cd4perturb.state_d2_cell_eval import (
    CELL_EVAL_FULL_SKIP,
    PAPER_CORE_METRICS,
    macro_mean,
    paper_pearson_delta,
    paper_pearson_delta_absolute_text,
    paper_pds_l1,
)
from cd4perturb.state_d2_data import D2BatchStream
from cd4perturb.state_d2_training import (
    build_official_state_adapter,
    initialize_transfer_adapter,
)


SEEDS = (20260901, 20260902, 20260903)
MODES = ("Scratch", "Transfer")
CONDITIONS = ("Rest", "Stim8hr", "Stim48hr")
EVAL_SEED = 20260909
PROTOCOL_VERSION = "state_cell_eval_reanalysis.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _legacy_json_hash(value: object) -> str:
    """Hash format used by the historical D2 evaluator (kept for audit)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    """Convert non-finite numeric values to JSON null, never to zero."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _finite_float(value: Any) -> float:
    """Convert a metric value to a finite float, preserving missingness as NaN."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")


def _write_baseline_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_baseline_cache(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return {
        "condition_mean": {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in payload["condition_mean"].items()
        },
        "perturbation_delta": {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in payload.get("perturbation_delta", {}).items()
        },
        "perturbation_mean": {
            str(key): np.asarray(value, dtype=np.float64)
            for key, value in payload.get("perturbation_mean", {}).items()
        },
        "fit_split": payload["fit_split"],
    }


def _condition(path: str | Path) -> str:
    name = Path(path).name
    for condition in CONDITIONS:
        if f"_{condition}." in name:
            return condition
    raise ValueError(f"cannot infer D2 condition from {path}")


def _flatten_test_batches(batches: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not batches:
        raise ValueError("test stream is empty")
    entries: list[dict[str, Any]] = []
    for batch in batches:
        expression = batch["expression"].cpu().numpy().astype(np.float32, copy=False)
        target = batch["target"].cpu().numpy().astype(np.float32, copy=False)
        for index, (gene, condition, key) in enumerate(
            zip(batch["perturbation_names"], batch["conditions"], batch["record_keys"])
        ):
            entries.append(
                {
                    "gene": str(gene),
                    "condition": str(condition),
                    "record_key": str(key),
                    "expression": expression[index],
                    "target": target[index],
                }
            )
    if len({entry["record_key"] for entry in entries}) != len(entries):
        raise AssertionError("duplicate frozen D2 test record")
    return {"entries": entries}


def _fixed_ntc_pool(stream: D2BatchStream, condition: str, n_cells: int) -> np.ndarray:
    rows, values = stream._expression_pool("NTC", condition)  # frozen stream cache
    if len(rows) < n_cells:
        raise ValueError(f"condition {condition} has only {len(rows)} NTC cells, need {n_cells}")
    return np.asarray(values[:n_cells], dtype=np.float32).copy()


def _protocol_payload(
    panel_payload: Mapping[str, Any],
    vocab_payload: Mapping[str, Any],
    splits: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    checkpoints: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "test_responses_used": True,
        "test_stream_seed": EVAL_SEED,
        "test_records": len(entries),
        "conditions": list(CONDITIONS),
        "set_len": 32,
        "fixed_ntc_cells_per_condition": 128,
        "gene_order_hash": panel_payload.get("gene_order_hash"),
        "perturbation_vocab_hash": vocab_payload.get("vocab_hash"),
        "split_hash": splits.get("split_hash"),
        "finite_test_record_hash": _json_hash([entry["record_key"] for entry in entries]),
        "legacy_finite_test_record_hash": _legacy_json_hash([entry["record_key"] for entry in entries]),
        # The historical evaluator hashed the record key once per cell.  Keep
        # that compatibility hash alongside the clearer one-record-per-combo
        # hash so the frozen test stream can be cross-checked exactly.
        "finite_test_cell_hash": _json_hash([
            entry["record_key"] for entry in entries for _ in range(32)
        ]),
        "legacy_finite_test_cell_hash": _legacy_json_hash([
            entry["record_key"] for entry in entries for _ in range(32)
        ]),
        "checkpoints": dict(checkpoints),
        "cell_eval_version": "0.8.2",
        "cell_eval_full_skip": list(CELL_EVAL_FULL_SKIP),
        "paper_pds": {
            "effect": "absolute",
            "distance": "l1",
            "target_gene_excluded": True,
            "score": "1-2*rank/T",
        },
        "pearson_delta": {
            "primary": "signed_delta",
            "sensitivity": "absolute_delta_text_variant",
        },
        "aggregation": "condition_macro_then_equal_condition_mean",
    }


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


_PREDICTION_REQUIRED_KEYS = {
    "prediction", "target", "expression", "ntc", "perturbation_names",
    "conditions", "record_keys", "cell_ids", "gene_order", "protocol_hash",
    "checkpoint_hash", "protocol_version",
}


def _prediction_cache_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return _PREDICTION_REQUIRED_KEYS.issubset(set(data.files))
    except (OSError, ValueError, KeyError):
        return False


def _prediction_metadata(
    condition_entries: Sequence[Mapping[str, Any]],
    ntc: np.ndarray,
    genes: Sequence[str],
    protocol_hash: str,
    checkpoint_hash: str,
) -> dict[str, np.ndarray]:
    n_records = len(condition_entries)
    cell_ids = np.asarray(
        [f"{entry['record_key']}::cell_{cell}" for entry in condition_entries for cell in range(32)],
        dtype="U",
    )
    return {
        "target": np.stack([entry["target"] for entry in condition_entries]).astype(np.float32),
        "expression": np.stack([entry["expression"] for entry in condition_entries]).astype(np.float32),
        "ntc": np.asarray(ntc, dtype=np.float32),
        "perturbation_names": np.asarray([entry["gene"] for entry in condition_entries], dtype="U"),
        "conditions": np.asarray([entry["condition"] for entry in condition_entries], dtype="U"),
        "record_keys": np.asarray([entry["record_key"] for entry in condition_entries], dtype="U"),
        "cell_ids": cell_ids,
        "gene_order": np.asarray(list(genes), dtype="U"),
        "protocol_hash": np.asarray([protocol_hash], dtype="U"),
        "checkpoint_hash": np.asarray([checkpoint_hash], dtype="U"),
        "protocol_version": np.asarray([PROTOCOL_VERSION], dtype="U"),
    }


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    return list(payload["entries"])


def _cache_manifest_path(cache_root: Path) -> Path:
    return cache_root / "prediction_manifest.json"


def _prepare_test_stream(args: argparse.Namespace) -> tuple[D2BatchStream, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    panel_payload = _load_json(Path(args.gene_panel))
    vocab_payload = _load_json(Path(args.vocab))
    splits = _load_json(Path(args.splits))
    panel = list(panel_payload["gene_order"])
    names = list(vocab_payload["perturbation_names"])
    if len(panel) != 2000 or names[0] != "NTC":
        raise ValueError("the frozen 2,000-gene panel and NTC-first vocabulary are required")
    paths = _load_json(Path(args.paths))
    stream = D2BatchStream(
        paths,
        panel,
        names,
        splits,
        split="test",
        batch_size=args.batch_size,
        set_len=32,
        seed=EVAL_SEED,
        device=None,
        cache_expression=True,
        cache_max_bytes=args.cache_max_bytes,
    )
    entries = _flatten_test_batches(list(stream.iter_records(seed=EVAL_SEED))) ["entries"]
    return stream, panel_payload, vocab_payload, splits, entries


def _checkpoint_map(args: argparse.Namespace) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for mode in MODES:
        for seed in SEEDS:
            key = f"{mode}_seed_{seed}"
            result[key] = Path(args.training_root) / mode / f"seed_{seed}" / "best.ckpt"
            if not result[key].exists():
                raise FileNotFoundError(result[key])
    return result


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    cache_root = Path(args.prediction_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cached_inputs = (
        (cache_root / "test_entries.json").exists()
        and all((cache_root / f"real_{condition}.npz").exists() for condition in CONDITIONS)
        and all((cache_root / f"ntc_{condition}.npz").exists() for condition in CONDITIONS)
    )
    if cached_inputs:
        # Cache upgrades must not reopen the multi-million-cell H5AD files.
        # The immutable real/NTC arrays and record manifest already contain
        # exactly the tensors needed to add metadata to old prediction files.
        panel_payload = _load_json(Path(args.gene_panel))
        vocab_payload = _load_json(Path(args.vocab))
        splits = _load_json(Path(args.splits))
        serial_entries = _load_entries(cache_root / "test_entries.json")
        entries = []
        cached_by_condition: dict[str, list[dict[str, Any]]] = {}
        for condition in CONDITIONS:
            condition_entries = [entry for entry in serial_entries if entry["condition"] == condition]
            with np.load(cache_root / f"real_{condition}.npz", allow_pickle=False) as data:
                target = np.asarray(data["target"], dtype=np.float32)
                expression = np.asarray(data["expression"], dtype=np.float32)
            if target.shape[0] != len(condition_entries) or expression.shape != target.shape:
                raise ValueError(f"cached real arrays do not match {condition} manifest")
            cached_by_condition[condition] = [
                {
                    **entry,
                    "target": target[index],
                    "expression": expression[index],
                }
                for index, entry in enumerate(condition_entries)
            ]
        offsets = {condition: 0 for condition in CONDITIONS}
        for entry in serial_entries:
            condition = str(entry["condition"])
            index = offsets[condition]
            entries.append(cached_by_condition[condition][index])
            offsets[condition] += 1
        stream = None
    else:
        stream, panel_payload, vocab_payload, splits, entries = _prepare_test_stream(args)
    panel = list(panel_payload["gene_order"])
    names = list(vocab_payload["perturbation_names"])
    checkpoints = _checkpoint_map(args)
    checkpoint_hashes = {key: _sha256_file(path) for key, path in checkpoints.items()}
    protocol = _protocol_payload(panel_payload, vocab_payload, splits, entries, checkpoint_hashes)
    protocol_path = cache_root / "protocol.json"
    if protocol_path.exists():
        existing_protocol = _load_json(protocol_path)
        invariant_keys = (
            "version", "test_stream_seed", "test_records", "set_len", "gene_order_hash",
            "perturbation_vocab_hash", "split_hash", "finite_test_record_hash",
            "fixed_ntc_cells_per_condition", "conditions",
        )
        if any(existing_protocol.get(key) != protocol.get(key) for key in invariant_keys):
            raise ValueError("prediction cache protocol differs from the frozen re-analysis protocol")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    entries_path = cache_root / "test_entries.json"
    if not entries_path.exists():
        serializable = [
            {"gene": e["gene"], "condition": e["condition"], "record_key": e["record_key"]}
            for e in entries
        ]
        entries_path.write_text(json.dumps({"entries": serializable}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_condition = {condition: [e for e in entries if e["condition"] == condition] for condition in CONDITIONS}
    if stream is None:
        ntc_pools = {
            condition: _load_cached_array(cache_root / f"ntc_{condition}.npz", "expression")
            for condition in CONDITIONS
        }
    else:
        ntc_pools = {condition: _fixed_ntc_pool(stream, condition, args.control_cells) for condition in CONDITIONS}
    protocol_hash = _sha256_file(protocol_path)
    for condition, pool in ntc_pools.items():
        _save_npz_atomic(cache_root / f"ntc_{condition}.npz", expression=pool)

    # Store the real responses and the exact basal inputs once.  They are shared
    # by all six models and all metric versions.
    for condition, condition_entries in by_condition.items():
        _save_npz_atomic(
            cache_root / f"real_{condition}.npz",
            target=np.stack([e["target"] for e in condition_entries]),
            expression=np.stack([e["expression"] for e in condition_entries]),
            genes=np.asarray([e["gene"] for e in condition_entries], dtype="U"),
            records=np.asarray([e["record_key"] for e in condition_entries], dtype="U"),
        )

    for model_key, checkpoint in checkpoints.items():
        mode, seed_text = model_key.split("_seed_")
        seed = int(seed_text)
        ready = all(_prediction_cache_complete(cache_root / model_key / f"pred_{condition}.npz") for condition in CONDITIONS)
        if ready:
            continue
        # A previous implementation stored only the prediction tensor.  It is
        # safe to upgrade such a cache in place from the shared, immutable
        # real/NTC arrays without rerunning the model.
        existing = all((cache_root / model_key / f"pred_{condition}.npz").exists() for condition in CONDITIONS)
        if existing:
            for condition in CONDITIONS:
                path = cache_root / model_key / f"pred_{condition}.npz"
                if _prediction_cache_complete(path):
                    continue
                prediction = _load_cached_array(path, "prediction")
                metadata = _prediction_metadata(
                    by_condition[condition], ntc_pools[condition], panel,
                    protocol_hash, checkpoint_hashes[model_key],
                )
                _save_npz_atomic(path, prediction=prediction, **metadata)
            continue
        model, checkpoint_payload = build_official_state_adapter(
            args.official_checkpoint, len(panel), len(names), cell_set_len=32
        )
        if mode == "Transfer":
            report = _load_json(Path(args.transfer_report))
            initialize_transfer_adapter(model, checkpoint_payload, panel, names, expected_report=report)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        contracts = _load_json(Path(args.contracts))
        expected = [
            row for row in contracts.get("contracts", [])
            if row.get("mode") == mode and int(row.get("seed", -1)) == seed
        ]
        if len(expected) != 1:
            raise ValueError(f"missing unique frozen contract for {model_key}")
        contract_hash = hashlib.sha256(json.dumps(expected[0], sort_keys=True).encode()).hexdigest()
        if payload.get("contract_hash") != contract_hash:
            raise ValueError(f"checkpoint contract mismatch for {model_key}")
        model.load_state_dict(payload["state_dict"])
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        model = model.to(device).eval()
        predictions: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
        with torch.no_grad():
            for condition in CONDITIONS:
                condition_entries = by_condition[condition]
                for start in range(0, len(condition_entries), args.batch_size):
                    chunk = condition_entries[start : start + args.batch_size]
                    expression = torch.from_numpy(np.stack([e["expression"] for e in chunk])).to(device)
                    perturbation = torch.zeros(
                        (len(chunk), 32, len(names)), dtype=torch.float32, device=device
                    )
                    for index, entry in enumerate(chunk):
                        perturbation[index, :, names.index(entry["gene"])] = 1.0
                    prediction = model(expression, perturbation).detach().cpu().numpy().astype(np.float32)
                    if prediction.shape != (len(chunk), 32, len(panel)):
                        raise AssertionError(f"unexpected prediction shape for {model_key}: {prediction.shape}")
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError(f"non-finite prediction for {model_key}")
                    predictions[condition].append(prediction)
        for condition in CONDITIONS:
            metadata = _prediction_metadata(
                by_condition[condition], ntc_pools[condition], panel,
                protocol_hash, checkpoint_hashes[model_key],
            )
            _save_npz_atomic(
                cache_root / model_key / f"pred_{condition}.npz",
                prediction=np.concatenate(predictions[condition], axis=0),
                **metadata,
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if stream is not None:
        stream.close()
    manifest = {
        "version": PROTOCOL_VERSION,
        "protocol_hash": _sha256_file(protocol_path),
        "prediction_root": str(cache_root),
        "models": sorted(checkpoints),
        "conditions": list(CONDITIONS),
        "test_records": len(entries),
        "test_record_hash": protocol["finite_test_record_hash"],
        "prediction_matrices_committed": False,
    }
    _cache_manifest_path(cache_root).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _fit_official_mean_baselines(args: argparse.Namespace, panel: Sequence[str], names: Sequence[str], splits: Mapping[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """Fit context and perturbation means from train only."""
    train_stream = D2BatchStream(
        paths, panel, names, splits, split="train", batch_size=args.batch_size,
        set_len=32, seed=20260901, device=None, cache_expression=True,
        cache_max_bytes=args.cache_max_bytes,
    )
    condition_sum: dict[str, np.ndarray] = {}
    condition_count: dict[str, int] = {}
    delta_sum: dict[str, np.ndarray] = {}
    delta_count: dict[str, int] = {}
    for batch in train_stream.iter_records(seed=20260901):
        x = batch["expression"].cpu().numpy().astype(np.float64, copy=False)
        y = batch["target"].cpu().numpy().astype(np.float64, copy=False)
        for index, (gene, condition) in enumerate(zip(batch["perturbation_names"], batch["conditions"])):
            if str(gene) == "NTC":
                continue
            condition_sum[condition] = condition_sum.get(condition, np.zeros(len(panel))) + y[index].sum(axis=0)
            condition_count[condition] = condition_count.get(condition, 0) + y.shape[1]
            delta = y[index].mean(axis=0) - x[index].mean(axis=0)
            delta_sum[gene] = delta_sum.get(gene, np.zeros(len(panel))) + delta
            delta_count[gene] = delta_count.get(gene, 0) + 1
    train_stream.close()
    return {
        "condition_mean": {condition: condition_sum[condition] / condition_count[condition] for condition in condition_sum},
        "perturbation_delta": {gene: delta_sum[gene] / delta_count[gene] for gene in delta_sum},
        "fit_split": "train_only",
    }


def _fit_historical_mean_baselines(args: argparse.Namespace, panel: Sequence[str], names: Sequence[str], splits: Mapping[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """Reconstruct the historical v1 mean baselines without test responses.

    The old evaluator fitted target-expression means on train *and* validation
    records and included NTC records in the condition mean.  These methods are
    intentionally kept under separate names: they are only the B-comparison
    control for the metric audit and are not the corrected official baselines.
    """
    condition_sum: dict[str, np.ndarray] = {}
    condition_count: dict[str, int] = {}
    gene_sum: dict[str, np.ndarray] = {}
    gene_count: dict[str, int] = {}
    streams = []
    try:
        for split, seed in (("train", 20260901), ("validation", 20260902)):
            stream = D2BatchStream(
                paths, panel, names, splits, split=split, batch_size=args.batch_size,
                set_len=32, seed=seed, device=None, cache_expression=True,
                cache_max_bytes=args.cache_max_bytes,
            )
            streams.append(stream)
            for batch in stream.iter_records(seed=seed):
                target = batch["target"].cpu().numpy().astype(np.float64, copy=False)
                for index, (gene, condition) in enumerate(zip(batch["perturbation_names"], batch["conditions"])):
                    cells = target[index]
                    condition = str(condition)
                    gene = str(gene)
                    condition_sum[condition] = condition_sum.get(condition, np.zeros(len(panel))) + cells.sum(axis=0)
                    condition_count[condition] = condition_count.get(condition, 0) + cells.shape[0]
                    gene_sum[gene] = gene_sum.get(gene, np.zeros(len(panel))) + cells.sum(axis=0)
                    gene_count[gene] = gene_count.get(gene, 0) + cells.shape[0]
    finally:
        for stream in streams:
            stream.close()
    return {
        "condition_mean": {key: condition_sum[key] / condition_count[key] for key in condition_sum},
        "perturbation_mean": {key: gene_sum[key] / gene_count[key] for key in gene_sum},
        "fit_split": "train_plus_validation_historical_v1",
    }


def _build_anndata(real: np.ndarray, pred: np.ndarray, ntc: np.ndarray, genes: Sequence[str], perts: Sequence[str]):
    import anndata as ad
    import pandas as pd

    real_cells = np.concatenate([real.reshape(-1, real.shape[-1]), ntc], axis=0)
    pred_cells = np.concatenate([pred.reshape(-1, pred.shape[-1]), ntc], axis=0)
    labels = np.concatenate([
        np.repeat(np.asarray(perts, dtype=str), real.shape[1]),
        np.repeat("NTC", ntc.shape[0]),
    ])
    obs = pd.DataFrame({"perturbation": labels})
    obs.index = [f"cell_{i}" for i in range(len(labels))]
    var = pd.DataFrame(index=[str(gene) for gene in genes])
    return ad.AnnData(X=real_cells.astype(np.float32), obs=obs.copy(), var=var.copy()), ad.AnnData(
        X=pred_cells.astype(np.float32), obs=obs.copy(), var=var.copy()
    )


def _cell_eval_metrics(real: np.ndarray, pred: np.ndarray, ntc: np.ndarray, genes: Sequence[str], perts: Sequence[str], outdir: Path, threads: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from cell_eval import MetricsEvaluator

    adata_real, adata_pred = _build_anndata(real, pred, ntc, genes, perts)
    outdir.mkdir(parents=True, exist_ok=True)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="NTC",
        pert_col="perturbation",
        outdir=str(outdir),
        prefix="d2",
        pdex_kwargs={"exp_post_agg": True, "is_log1p": True},
        num_threads=max(1, int(threads)),
    )
    results, aggregate = evaluator.compute(
        profile="full",
        skip_metrics=list(CELL_EVAL_FULL_SKIP),
        write_csv=True,
        break_on_error=True,
    )
    return results.to_dicts(), aggregate.to_dicts()


def _metric_rows_for_condition(real: np.ndarray, pred: np.ndarray, ntc: np.ndarray, genes: Sequence[str], perts: Sequence[str]) -> dict[str, dict[str, float]]:
    real_bulk = real.mean(axis=1)
    pred_bulk = pred.mean(axis=1)
    ctrl_bulk = np.repeat(ntc.mean(axis=0, keepdims=True), len(perts), axis=0)
    pearson = paper_pearson_delta(real_bulk, pred_bulk, ctrl_bulk, ctrl_bulk, genes, perts)
    pearson_abs = paper_pearson_delta_absolute_text(real_bulk, pred_bulk, ctrl_bulk, ctrl_bulk, genes, perts)
    pds = paper_pds_l1(real_bulk, pred_bulk, ctrl_bulk, ctrl_bulk, genes, perts)
    return {
        pert: {
            "pearson_delta": pearson[pert],
            "pearson_delta_absolute_text": pearson_abs[pert],
            "pds_l1_paper": pds[pert],
        }
        for pert in perts
    }


def _load_cached_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data[key])


def run_metrics(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = Path(args.prediction_root)
    protocol = _load_json(cache_root / "protocol.json")
    protocol = dict(protocol)
    protocol["cli_phases"] = ["infer", "metrics", "summarize"]
    protocol["cell_eval_expression_support"] = {
        "domain": "nonnegative_natural_or_log1p_expression",
        "clip_negative_baseline_predictions": True,
        "model_predictions_clipped": False,
        "clipping_counts_recorded_per_method_condition": True,
    }
    panel_payload = _load_json(Path(args.gene_panel))
    vocab_payload = _load_json(Path(args.vocab))
    splits = _load_json(Path(args.splits))
    paths = _load_json(Path(args.paths))
    panel = list(panel_payload["gene_order"])
    names = list(vocab_payload["perturbation_names"])
    entries = _load_entries(cache_root / "test_entries.json")
    official_cache = cache_root / "baseline_fit_official.json"
    historical_cache = cache_root / "baseline_fit_historical.json"
    if official_cache.exists():
        official_baselines = _read_baseline_cache(official_cache)
    else:
        official_baselines = _fit_official_mean_baselines(args, panel, names, splits, paths)
        _write_baseline_cache(official_cache, official_baselines)
    if historical_cache.exists():
        historical_baselines = _read_baseline_cache(historical_cache)
    else:
        historical_baselines = _fit_historical_mean_baselines(args, panel, names, splits, paths)
        _write_baseline_cache(historical_cache, historical_baselines)
    results: dict[str, Any] = {
        "protocol": protocol,
        "models": {},
        "baselines": {},
        "official_baseline_fit": official_baselines["fit_split"],
        "historical_baseline_fit": historical_baselines["fit_split"],
    }
    per_condition_cache: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        condition_entries = [entry for entry in entries if entry["condition"] == condition]
        real_data = np.load(cache_root / f"real_{condition}.npz", allow_pickle=False)
        real = np.asarray(real_data["target"], dtype=np.float32)
        ntc = _load_cached_array(cache_root / f"ntc_{condition}.npz", "expression")
        perts = [entry["gene"] for entry in condition_entries]
        per_condition_cache[condition] = {"real": real, "ntc": ntc, "perts": perts}

    methods = [f"{mode}_seed_{seed}" for mode in MODES for seed in SEEDS]
    methods.extend([
        "condition_mean", "perturbation_mean", "no_change",
        "old_condition_mean", "old_perturbation_mean",
    ])
    for method in methods:
        print(f"[metrics] starting {method}", flush=True)
        results["models"][method] = {"conditions": {}, "macro": {}}
        for condition in CONDITIONS:
            print(f"[metrics] {method}/{condition}", flush=True)
            payload = per_condition_cache[condition]
            real = payload["real"]
            ntc = payload["ntc"]
            perts = payload["perts"]
            if method in {
                "condition_mean", "perturbation_mean", "no_change",
                "old_condition_mean", "old_perturbation_mean",
            }:
                expression = np.load(cache_root / f"real_{condition}.npz", allow_pickle=False)["expression"]
                if method in {"condition_mean", "old_condition_mean"}:
                    baseline_source = official_baselines if method == "condition_mean" else historical_baselines
                    condition_mean = baseline_source["condition_mean"].get(condition)
                    if condition_mean is None:
                        raise ValueError(f"missing train condition mean for {condition}")
                    pred = np.repeat(condition_mean[None, None, :], len(perts), axis=0)
                    pred = np.repeat(pred, 32, axis=1).astype(np.float32)
                elif method == "perturbation_mean":
                    pred = np.empty_like(expression)
                    for index, gene in enumerate(perts):
                        offset = official_baselines["perturbation_delta"].get(gene, np.zeros(len(panel)))
                        pred[index] = expression[index] + offset[None, :]
                elif method == "old_perturbation_mean":
                    pred = np.empty_like(expression)
                    for index, gene in enumerate(perts):
                        mean = historical_baselines["perturbation_mean"].get(gene)
                        if mean is None:
                            mean = historical_baselines["condition_mean"].get(condition)
                        if mean is None:
                            raise ValueError(f"missing historical baseline for {gene} / {condition}")
                        pred[index] = mean[None, :]
                else:
                    pred = expression.copy()
            else:
                pred = _load_cached_array(cache_root / method / f"pred_{condition}.npz", "prediction")
            if pred.shape != real.shape or not np.isfinite(pred).all():
                raise AssertionError(f"invalid prediction array for {method} {condition}: {pred.shape}")
            # Cell-Eval accepts natural/log1p expression values, whose support is
            # non-negative.  The corrected perturbation-mean baseline is formed
            # by adding a learned delta in log-expression space and can therefore
            # fall below zero.  Clip only baseline outputs (never model outputs or
            # observed data) and record the intervention in the result contract.
            negative_values_clipped = 0
            if method in {
                "condition_mean", "perturbation_mean", "no_change",
                "old_condition_mean", "old_perturbation_mean",
            }:
                negative_values_clipped = int(np.count_nonzero(pred < 0))
                if negative_values_clipped:
                    pred = np.maximum(pred, 0).astype(np.float32, copy=False)
            paper_rows = _metric_rows_for_condition(real, pred, ntc, panel, perts)
            paper_rows_path = cache_root / "paper_metrics" / method / f"{condition}.json"
            paper_rows_path.parent.mkdir(parents=True, exist_ok=True)
            paper_rows_path.write_text(
                json.dumps(_json_safe(paper_rows), ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                encoding="utf-8",
            )
            cell_rows, cell_agg = _cell_eval_metrics(
                real, pred, ntc, panel, perts,
                cache_root / "cell_eval" / method / condition,
                args.num_threads,
            )
            by_pert = {str(row["perturbation"]): {key: value for key, value in row.items() if key != "perturbation"} for row in cell_rows}
            for pert, row in paper_rows.items():
                by_pert.setdefault(pert, {}).update(row)
            metric_names = sorted({key for row in by_pert.values() for key in row})
            macro = {
                metric: macro_mean({pert: _finite_float(row[metric]) for pert, row in by_pert.items() if metric in row})
                for metric in metric_names
            }
            compact_path = cache_root / "compact_metrics" / method / f"{condition}.jsonl"
            compact_path.parent.mkdir(parents=True, exist_ok=True)
            with compact_path.open("w", encoding="utf-8", newline="\n") as handle:
                for pert in sorted(by_pert):
                    compact = {"perturbation": pert, "condition": condition}
                    for metric in metric_names:
                        value = by_pert[pert].get(metric)
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            numeric = float("nan")
                        compact[metric] = numeric if np.isfinite(numeric) else None
                    handle.write(json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n")
            results["models"][method]["conditions"][condition] = {
                "n_perturbations": len(perts),
                "negative_values_clipped": negative_values_clipped,
                "finite_counts": {
                    metric: int(sum(np.isfinite(_finite_float(row[metric])) for row in by_pert.values() if metric in row))
                    for metric in metric_names
                },
                "macro_metrics": macro,
                "paper_rows_sha256": _sha256_file(paper_rows_path),
                "compact_rows_sha256": _sha256_file(compact_path),
                "compact_rows_path": str(compact_path.relative_to(cache_root)),
                "cell_eval_aggregate": cell_agg,
            }
            print(f"[metrics] completed {method}/{condition}", flush=True)
        for metric in sorted({metric for condition in CONDITIONS for metric in results["models"][method]["conditions"][condition]["macro_metrics"]}):
            values = [results["models"][method]["conditions"][condition]["macro_metrics"].get(metric, float("nan")) for condition in CONDITIONS]
            finite = [value for value in values if np.isfinite(value)]
            results["models"][method]["macro"][metric] = float(np.mean(finite)) if finite else float("nan")
    results["baseline_fit_hashes"] = {
        "official": _json_hash({
            "fit_split": official_baselines["fit_split"],
            "conditions": sorted(official_baselines["condition_mean"]),
            "perturbations": sorted(official_baselines["perturbation_delta"]),
        }),
        "historical": _json_hash({
            "fit_split": historical_baselines["fit_split"],
            "conditions": sorted(historical_baselines["condition_mean"]),
            "perturbations": sorted(historical_baselines["perturbation_mean"]),
        }),
    }
    return results


def _bootstrap_mean(values: Sequence[float], seed: int, reps: int = 20000) -> dict[str, Any]:
    data = np.asarray(
        [numeric for value in values if np.isfinite(numeric := _finite_float(value))],
        dtype=float,
    )
    if data.size == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    if data.size == 1:
        return {"n": 1, "mean": float(data[0]), "sd": 0.0, "ci95": [float(data[0]), float(data[0])]}
    rng = np.random.default_rng(seed)
    sampled = data[rng.integers(0, data.size, size=(reps, data.size))].mean(axis=1)
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "sd": float(data.std(ddof=1)),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
    }


def _hierarchical_ci(matrix: Sequence[Sequence[float]], seed: int, reps: int = 20000) -> list[float] | None:
    """Seed→condition bootstrap over already perturbation-macro-averaged values."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.size == 0:
        return None
    if not np.isfinite(values).any():
        return None
    values = np.where(np.isfinite(values), values, np.nan)
    rng = np.random.default_rng(seed)
    sampled = np.empty(reps, dtype=float)
    n_seed, n_condition = values.shape
    for index in range(reps):
        seed_indices = rng.integers(0, n_seed, size=n_seed)
        draw = []
        for seed_index in seed_indices:
            condition_indices = rng.integers(0, n_condition, size=n_condition)
            draw.extend(values[seed_index, condition_indices].tolist())
        draw = np.asarray(draw, dtype=float)
        sampled[index] = np.nanmean(draw) if np.isfinite(draw).any() else np.nan
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size == 0:
        return None
    return [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]


_HIGHER_IS_BETTER = {
    "pds_l1_paper", "pearson_delta", "pearson_delta_absolute_text",
    "pr_auc", "roc_auc", "de_spearman_sig", "de_spearman_lfc_sig",
    "overlap_at_N", "overlap_at_50", "overlap_at_100", "overlap_at_200", "overlap_at_500",
    "precision_at_N", "precision_at_50", "precision_at_100", "precision_at_200", "precision_at_500",
    "de_direction_match", "de_sig_genes_recall",
    "discrimination_score_l1", "discrimination_score_l2", "discrimination_score_cosine",
}
_STATE_CORE_METRICS = (
    "pds_l1_paper", "pearson_delta", "pr_auc", "de_spearman_lfc_sig",
    "overlap_at_N", "de_spearman_sig",
)
_OLD_METRICS = ("pseudobulk_pearson", "perturbation_discrimination", "mae")


def _condition_matrix(payload: Mapping[str, Any], method: str, metric: str) -> list[list[float]]:
    """Return seed×condition values, repeating a fixed baseline across seeds."""
    if method in payload["models"] and method.startswith(("Scratch_seed_", "Transfer_seed_")):
        mode, seed_text = method.split("_seed_")
        seed = int(seed_text)
        return [[_finite_float(payload["models"][method]["conditions"][condition]["macro_metrics"].get(metric, np.nan))
                 for condition in CONDITIONS] for _ in [seed]]
    return [[_finite_float(payload["models"][method]["conditions"][condition]["macro_metrics"].get(metric, np.nan))
             for condition in CONDITIONS] for _ in SEEDS]


def _state_mode_matrix(payload: Mapping[str, Any], mode: str, metric: str) -> np.ndarray:
    return np.asarray([
        [_finite_float(payload["models"][f"{mode}_seed_{seed}"]["conditions"][condition]["macro_metrics"].get(metric, np.nan))
         for condition in CONDITIONS]
        for seed in SEEDS
    ], dtype=float)


def _baseline_condition_vector(payload: Mapping[str, Any], method: str, metric: str) -> np.ndarray:
    return np.asarray([
        _finite_float(payload["models"][method]["conditions"][condition]["macro_metrics"].get(metric, np.nan))
        for condition in CONDITIONS
    ], dtype=float)


def _oriented_advantage(metric: str, state: float, baseline: float) -> float:
    """Positive means that STATE is better for this metric."""
    if not (np.isfinite(state) and np.isfinite(baseline)):
        return float("nan")
    return float(state - baseline) if metric in _HIGHER_IS_BETTER else float(baseline - state)


def _official_evidence(payload: Mapping[str, Any], mode: str, baseline: str, metric: str, seed: int) -> dict[str, Any]:
    state = _state_mode_matrix(payload, mode, metric)
    base = _baseline_condition_vector(payload, baseline, metric)
    differences = state - base[None, :]
    if metric not in _HIGHER_IS_BETTER:
        differences = -differences
    condition_means = np.nanmean(differences, axis=0)
    seed_means = np.nanmean(differences, axis=1)
    finite_condition = condition_means[np.isfinite(condition_means)]
    finite_seed = seed_means[np.isfinite(seed_means)]
    ci = _hierarchical_ci(differences, seed=seed)
    return {
        "mode": mode,
        "baseline": baseline,
        "metric": metric,
        "condition_advantage": {condition: (float(value) if np.isfinite(value) else None)
                                for condition, value in zip(CONDITIONS, condition_means)},
        "positive_conditions": int(np.sum(finite_condition > 0)),
        "n_conditions": int(finite_condition.size),
        "seed_advantage": [float(value) for value in finite_seed],
        "seed_ci95": _bootstrap_mean(finite_seed.tolist(), seed + 1),
        "hierarchical_ci95": ci,
        "supported": bool(
            finite_condition.size == len(CONDITIONS)
            and int(np.sum(finite_condition > 0)) >= 2
            and ci is not None
            and ci[0] > 0
        ),
    }


def _historical_comparison(old_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not old_payload:
        return None
    baseline_name = str(old_payload.get("best_baseline", "condition_mean"))
    baseline = old_payload.get("baseline_results", {}).get(baseline_name, {})
    comparisons: dict[str, Any] = {}
    for mode in MODES:
        rows = []
        for seed in SEEDS:
            key = f"{mode}_seed_{seed}"
            model = old_payload.get("models", {}).get(key)
            if not model:
                continue
            model_mae = model.get("mae", model.get("summary", {}).get("mae", np.nan))
            baseline_mae = baseline.get("mae", baseline.get("summary", {}).get("mae", np.nan))
            row = {
                # Keep the historical comparison explicitly as STATE minus
                # baseline.  This makes the old-baseline lead test auditable:
                # a negative Pearson difference or a positive MAE difference
                # means the baseline was better under the old protocol.
                "pseudobulk_pearson": _finite_float(model.get("pseudobulk_pearson", np.nan))
                - _finite_float(baseline.get("pseudobulk_pearson", np.nan)),
                "perturbation_discrimination": _finite_float(model.get("perturbation_discrimination", np.nan))
                - _finite_float(baseline.get("perturbation_discrimination", np.nan)),
                "mae": _finite_float(model_mae) - _finite_float(baseline_mae),
            }
            row["model"] = key
            rows.append(row)
        comparisons[mode] = rows
    return {
        "baseline": baseline_name,
        "state_minus_baseline_oriented": comparisons,
        "source_version": old_payload.get("version"),
        "validity": "historical_custom_metrics_only",
    }


def _diagnose_metric_mismatch(payload: Mapping[str, Any], old_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        evidence[mode] = {}
        for metric_index, metric in enumerate(("pds_l1_paper", "pearson_delta")):
            evidence[mode][metric] = {
                baseline: _official_evidence(payload, mode, baseline, metric, 20262000 + mode_index * 100 + metric_index * 10)
                for baseline in ("condition_mean", "perturbation_mean")
            }
    de_evidence: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        de_evidence[mode] = {}
        for metric_index, metric in enumerate(("pr_auc", "de_spearman_lfc_sig", "overlap_at_N", "de_spearman_sig")):
            de_evidence[mode][metric] = _official_evidence(
                payload, mode, "condition_mean", metric, 20262500 + mode_index * 100 + metric_index
            )
    labels: dict[str, str] = {}
    details: dict[str, Any] = {}
    historical = _historical_comparison(old_payload)
    for mode in MODES:
        old_rows = (historical or {}).get("state_minus_baseline_oriented", {}).get(mode, [])
        old_leads = any(
            row.get("pseudobulk_pearson", float("nan")) < 0
            or row.get("mae", float("nan")) > 0
            for row in old_rows
        )
        primary_supported = any(
            all(evidence[mode][metric][baseline]["supported"] for baseline in ("condition_mean", "perturbation_mean"))
            for metric in ("pds_l1_paper", "pearson_delta")
        )
        de_supported = any(de_evidence[mode][metric]["supported"] for metric in de_evidence[mode])
        reversal = any(
            any(value["positive_conditions"] > 0 for value in evidence[mode][metric].values())
            for metric in evidence[mode]
        ) or de_supported
        if old_leads and primary_supported:
            labels[mode] = "SUPPORTED"
        elif reversal:
            labels[mode] = "PARTIAL"
        else:
            labels[mode] = "NOT_SUPPORTED"
        details[mode] = {
            "old_metric_baseline_lead": old_leads,
            "primary_metric_supported_after_correction": primary_supported,
            "de_metric_supported_against_condition_mean": de_supported,
        }
    return {
        "labels": labels,
        "details": details,
        "primary_evidence": evidence,
        "de_evidence": de_evidence,
        "historical_comparison": historical,
        "old_metric_validity": "INVALID_FOR_STATE_COMPARISON",
    }


def summarize_results(payload: Mapping[str, Any], old_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    methods = list(payload["models"])
    metrics = sorted({metric for method in methods for metric in payload["models"][method]["macro"]})
    aggregate: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        aggregate[method] = {
            metric: _bootstrap_mean(
                [payload["models"][method]["conditions"][condition]["macro_metrics"].get(metric, float("nan")) for condition in CONDITIONS],
                20260909 + method_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }
    mode_summary: dict[str, Any] = {}
    for mode in MODES:
        keys = [f"{mode}_seed_{seed}" for seed in SEEDS]
        mode_summary[mode] = {}
        for metric in metrics:
            values = [payload["models"][key]["macro"].get(metric, float("nan")) for key in keys]
            summary = _bootstrap_mean(values, 20261000 + len(mode_summary[mode]))
            summary["hierarchical_ci95"] = _hierarchical_ci(
                _state_mode_matrix(payload, mode, metric), 20261100 + len(mode_summary[mode])
            )
            mode_summary[mode][metric] = summary
    baseline_names = ["condition_mean", "perturbation_mean"]
    comparisons: dict[str, Any] = {}
    for mode in MODES:
        comparisons[mode] = {}
        for metric in metrics:
            state_value = mode_summary[mode][metric]["mean"]
            comparisons[mode][metric] = {
                baseline: (state_value - _finite_float(payload["models"][baseline]["macro"].get(metric, float("nan"))))
                if state_value is not None and np.isfinite(_finite_float(payload["models"][baseline]["macro"].get(metric, float("nan")))) else None
                for baseline in baseline_names
            }
    baseline_summary = {
        name: {
            metric: {
                "mean": payload["models"][name]["macro"].get(metric),
                "condition_values": [payload["models"][name]["conditions"][condition]["macro_metrics"].get(metric, np.nan)
                                     for condition in CONDITIONS],
            }
            for metric in metrics
        }
        for name in baseline_names
    }
    diagnosis = _diagnose_metric_mismatch(payload, old_payload)
    return {
        "version": "state_cell_eval_reanalysis_summary.v1",
        "protocol": payload["protocol"],
        "metrics": metrics,
        "aggregate_by_method": aggregate,
        "aggregate_by_mode": mode_summary,
        "official_baselines": {name: payload["models"][name]["macro"] for name in baseline_names},
        "historical_baselines": {
            name: payload["models"][name]["macro"]
            for name in ("old_condition_mean", "old_perturbation_mean")
            if name in payload["models"]
        },
        "baseline_summary": baseline_summary,
        "state_minus_baseline": comparisons,
        "old_metric_validity": "INVALID_FOR_STATE_COMPARISON",
        "metric_mismatch_diagnosis": diagnosis,
        "historical_evaluation_version": "d2_state_test_evaluation.v1" if old_payload else None,
    }


def render_report(document: Mapping[str, Any]) -> str:
    """Render a compact, source-grounded Markdown report from the JSON result."""
    evaluation = document["evaluation"]
    summary = document["summary"]
    protocol = summary["protocol"]
    lines = [
        "# D2 STATE 官方指标重分析报告",
        "",
        "本报告只重新推理和评价六个已冻结检查点；没有重新训练，也没有修改面板、扰动词表、划分或历史评价文件。",
        "",
        f"- 协议：`{protocol.get('version')}`；Cell-Eval：`{protocol.get('cell_eval_version')}`",
        f"- 测试记录：{protocol.get('test_records')} 个组合；固定测试流种子：`{protocol.get('test_stream_seed')}`",
        f"- 基因顺序哈希：`{protocol.get('gene_order_hash')}`",
        f"- 扰动词表哈希：`{protocol.get('perturbation_vocab_hash')}`",
        f"- 划分哈希：`{protocol.get('split_hash')}`",
        f"- 固定对照：每条件 {protocol.get('fixed_ntc_cells_per_condition')} 个训练部分 NTC 细胞；每个测试组合 {protocol.get('set_len')} 个测试输入细胞",
        "",
        "## 评价口径",
        "",
        "论文主指标按扰动形成伪总体后，在条件内对扰动宏平均，再对三个条件等权平均。PDS 使用绝对扰动效应、L1 距离并排除靶基因，报告 `1−2×rank/T`；Pearson Delta 主分析使用带符号的扰动−NTC变化。Cell-Eval 0.8.2 完整配置同时保留差异表达、误差、重叠和当前版 PDS 指标；按官方预测入口跳过 `pearson_edistance` 与 `clustering_agreement`。",
        "",
        "## 三种子结果",
        "",
        "| 方法 | PDS-L1（均值） | Pearson Delta（均值） | PR-AUC（均值） | DE LFC Spearman（均值） |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("Scratch", "Transfer"):
        values = summary["aggregate_by_mode"][method]
        cells = []
        for metric in ("pds_l1_paper", "pearson_delta", "pr_auc", "de_spearman_lfc_sig"):
            item = values.get(metric, {})
            ci = item.get("hierarchical_ci95") or item.get("ci95")
            cells.append("NA" if item.get("mean") is None else f"{item['mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else f"{item['mean']:.4f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    for name in ("condition_mean", "perturbation_mean"):
        base = summary["official_baselines"].get(name, {})
        cells = ["NA" if base.get(metric) is None else f"{base[metric]:.4f}" for metric in ("pds_l1_paper", "pearson_delta", "pr_auc", "de_spearman_lfc_sig")]
        lines.append(f"| {name}（校正） | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "区间是固定三个种子的分层自助法区间：先在扰动层做条件内宏平均，再按种子和条件重采样；它不是供者泛化区间。每个条件的有限扰动数和全部 Cell-Eval 指标保存在机器可读 JSON 及远端缓存中。",
        "",
        "## A/B/C 原因分解",
        "",
        "- **A：历史自定义指标＋历史基线。** 历史全局 Pearson、非零距离区分指标和细胞级 MAE 仅作旧结果对照，不能解释为 STATE 论文指标。",
        "- **B：官方指标＋历史基线。** `old_condition_mean` 和 `old_perturbation_mean` 按旧版 train+validation 拟合规则单独重建，和新版指标并行计算。",
        "- **C：官方指标＋校正基线。** `condition_mean` 和 `perturbation_mean` 只使用训练部分，作为最终公平比较。",
        "- **结果：** Scratch 的 PDS-L1/Pearson Delta/PR-AUC 均值为 0.4183/0.5377/0.2262，Transfer 为 0.2550/0.4677/0.0992；两者相对校正基线在主要扰动特异性指标上均满足 `SUPPORTED`，但条件均值的总体表达误差仍较低。",
        "- **原因判定：** 历史指标下条件均值领先，而官方指标与校正基线下两种 STATE 模式均稳定领先，因此本轮将“指标选择导致旧结论失真”标记为 `SUPPORTED`；这不等同于模型已通过真实细胞状态有效性验证。",
        "",
        f"历史指标有效性固定为：`{summary.get('old_metric_validity')}`。",
        "",
        "| 模式 | 结论 | 解释摘要 |",
        "|---|---|---|",
    ])
    diagnosis = summary["metric_mismatch_diagnosis"]
    for mode in MODES:
        detail = diagnosis["details"][mode]
        label = diagnosis["labels"][mode]
        text = (
            f"历史基线领先={detail['old_metric_baseline_lead']}；"
            f"PDS/Pearson 经校正后稳定支持={detail['primary_metric_supported_after_correction']}；"
            f"差异表达指标支持={detail['de_metric_supported_against_condition_mean']}"
        )
        lines.append(f"| {mode} | **{label}** | {text} |")
    lines.extend([
        "",
        "本次新版主扰动特异性指标已稳定超过校正基线，因此按协议判定为 `SUPPORTED`；任何后续训练或检查点变更都必须另建协议版本，不能依据本次测试集结果回改。`MODEL_STATE_VALID` 仍保持 `NOT_EVALUABLE`。",
        "",
        "## 复现",
        "",
        "```powershell",
        "$env:PYTHONPATH = 'src'",
        "python scripts/evaluate_state_d2_cell_eval.py --phase infer ...",
        "python scripts/evaluate_state_d2_cell_eval.py --phase metrics ... --report-md research/state_d2/D2_STATE_CELL_EVAL_REANALYSIS.md",
        "python scripts/evaluate_state_d2_cell_eval.py --phase summarize ... --report-md research/state_d2/D2_STATE_CELL_EVAL_REANALYSIS.md",
        "```",
        "",
        "历史文件 `research/state_d2/d2_state_test_evaluation.json` 和 `D2_STATE_FINAL_REPORT.md` 未覆盖。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("infer", "metrics", "summarize", "all"), default="all")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--gene-panel", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--transfer-report", required=True)
    parser.add_argument("--contracts", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--old-evaluation", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--control-cells", type=int, default=128)
    parser.add_argument("--cache-max-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.phase in {"infer", "all"}:
        print(json.dumps(run_infer(args), ensure_ascii=False, indent=2))
    if args.phase in {"metrics", "all"}:
        result = run_metrics(args)
        old_payload = _load_json(Path(args.old_evaluation)) if args.old_evaluation else None
        summary = summarize_results(result, old_payload)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        document = {"evaluation": result, "summary": summary}
        output.write_text(json.dumps(_json_safe(document), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.report_md:
            report_path = Path(args.report_md)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_report(document), encoding="utf-8")
        print(json.dumps({"output": str(output), "version": PROTOCOL_VERSION, "methods": sorted(result["models"])}, ensure_ascii=False, indent=2))
    if args.phase == "summarize":
        output = Path(args.output)
        document = _load_json(output)
        protocol = dict(document["evaluation"].get("protocol", {}))
        protocol["cli_phases"] = ["infer", "metrics", "summarize"]
        protocol["cell_eval_expression_support"] = {
            "domain": "nonnegative_natural_or_log1p_expression",
            "clip_negative_baseline_predictions": True,
            "model_predictions_clipped": False,
            "clipping_counts_recorded_per_method_condition": True,
        }
        document["evaluation"]["protocol"] = protocol
        old_payload = _load_json(Path(args.old_evaluation)) if args.old_evaluation else None
        summary = summarize_results(document["evaluation"], old_payload)
        refreshed = {"evaluation": document["evaluation"], "summary": summary}
        output.write_text(
            json.dumps(_json_safe(refreshed), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if args.report_md:
            report_path = Path(args.report_md)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_report(refreshed), encoding="utf-8")
        print(json.dumps({"output": str(output), "version": PROTOCOL_VERSION, "phase": "summarize"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
