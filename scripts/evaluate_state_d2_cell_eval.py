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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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

    stream, panel_payload, vocab_payload, splits, entries = _prepare_test_stream(args)
    panel = list(panel_payload["gene_order"])
    names = list(vocab_payload["perturbation_names"])
    cache_root = Path(args.prediction_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_map(args)
    checkpoint_hashes = {key: _sha256_file(path) for key, path in checkpoints.items()}
    protocol = _protocol_payload(panel_payload, vocab_payload, splits, entries, checkpoint_hashes)
    protocol_path = cache_root / "protocol.json"
    if protocol_path.exists() and _load_json(protocol_path) != protocol:
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
    ntc_pools = {condition: _fixed_ntc_pool(stream, condition, args.control_cells) for condition in CONDITIONS}
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
        ready = all((cache_root / model_key / f"pred_{condition}.npz").exists() for condition in CONDITIONS)
        if ready:
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
            _save_npz_atomic(
                cache_root / model_key / f"pred_{condition}.npz",
                prediction=np.concatenate(predictions[condition], axis=0),
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
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
    panel_payload = _load_json(Path(args.gene_panel))
    vocab_payload = _load_json(Path(args.vocab))
    splits = _load_json(Path(args.splits))
    paths = _load_json(Path(args.paths))
    panel = list(panel_payload["gene_order"])
    names = list(vocab_payload["perturbation_names"])
    entries = _load_entries(cache_root / "test_entries.json")
    official_baselines = _fit_official_mean_baselines(args, panel, names, splits, paths)
    results: dict[str, Any] = {"protocol": protocol, "models": {}, "baselines": {}, "official_baseline_fit": official_baselines["fit_split"]}
    per_condition_cache: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        condition_entries = [entry for entry in entries if entry["condition"] == condition]
        real_data = np.load(cache_root / f"real_{condition}.npz", allow_pickle=False)
        real = np.asarray(real_data["target"], dtype=np.float32)
        ntc = _load_cached_array(cache_root / f"ntc_{condition}.npz", "expression")
        perts = [entry["gene"] for entry in condition_entries]
        per_condition_cache[condition] = {"real": real, "ntc": ntc, "perts": perts}

    methods = [f"{mode}_seed_{seed}" for mode in MODES for seed in SEEDS]
    methods.extend(["condition_mean", "perturbation_mean", "no_change"])
    for method in methods:
        results["models"][method] = {"conditions": {}, "macro": {}}
        for condition in CONDITIONS:
            payload = per_condition_cache[condition]
            real = payload["real"]
            ntc = payload["ntc"]
            perts = payload["perts"]
            if method in {"condition_mean", "perturbation_mean", "no_change"}:
                expression = np.load(cache_root / f"real_{condition}.npz", allow_pickle=False)["expression"]
                if method == "condition_mean":
                    condition_mean = official_baselines["condition_mean"].get(condition)
                    if condition_mean is None:
                        raise ValueError(f"missing train condition mean for {condition}")
                    pred = np.repeat(condition_mean[None, None, :], len(perts), axis=0)
                    pred = np.repeat(pred, 32, axis=1).astype(np.float32)
                elif method == "perturbation_mean":
                    pred = np.empty_like(expression)
                    for index, gene in enumerate(perts):
                        offset = official_baselines["perturbation_delta"].get(gene, np.zeros(len(panel)))
                        pred[index] = expression[index] + offset[None, :]
                else:
                    pred = expression.copy()
            else:
                pred = _load_cached_array(cache_root / method / f"pred_{condition}.npz", "prediction")
            if pred.shape != real.shape or not np.isfinite(pred).all():
                raise AssertionError(f"invalid prediction array for {method} {condition}: {pred.shape}")
            paper_rows = _metric_rows_for_condition(real, pred, ntc, panel, perts)
            paper_rows_path = cache_root / "paper_metrics" / method / f"{condition}.json"
            paper_rows_path.parent.mkdir(parents=True, exist_ok=True)
            paper_rows_path.write_text(
                json.dumps(paper_rows, ensure_ascii=False, allow_nan=True, indent=2) + "\n",
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
                metric: macro_mean({pert: float(row[metric]) for pert, row in by_pert.items() if metric in row})
                for metric in metric_names
            }
            results["models"][method]["conditions"][condition] = {
                "n_perturbations": len(perts),
                "finite_counts": {
                    metric: int(sum(np.isfinite(float(row[metric])) for row in by_pert.values() if metric in row))
                    for metric in metric_names
                },
                "macro_metrics": macro,
                "paper_rows_sha256": _sha256_file(paper_rows_path),
                "cell_eval_aggregate": cell_agg,
            }
        for metric in sorted({metric for condition in CONDITIONS for metric in results["models"][method]["conditions"][condition]["macro_metrics"]}):
            values = [results["models"][method]["conditions"][condition]["macro_metrics"].get(metric, float("nan")) for condition in CONDITIONS]
            finite = [value for value in values if np.isfinite(value)]
            results["models"][method]["macro"][metric] = float(np.mean(finite)) if finite else float("nan")
    return results


def _bootstrap_mean(values: Sequence[float], seed: int, reps: int = 20000) -> dict[str, Any]:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
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
    state_methods = [f"{mode}_seed_{seed}" for mode in MODES for seed in SEEDS]
    mode_summary: dict[str, Any] = {}
    for mode in MODES:
        keys = [f"{mode}_seed_{seed}" for seed in SEEDS]
        mode_summary[mode] = {}
        for metric in metrics:
            values = [payload["models"][key]["macro"].get(metric, float("nan")) for key in keys]
            mode_summary[mode][metric] = _bootstrap_mean(values, 20261000 + len(mode_summary[mode]))
    baseline_names = ["condition_mean", "perturbation_mean"]
    comparisons: dict[str, Any] = {}
    for mode in MODES:
        comparisons[mode] = {}
        for metric in metrics:
            state_value = mode_summary[mode][metric]["mean"]
            comparisons[mode][metric] = {
                baseline: (state_value - payload["models"][baseline]["macro"].get(metric, float("nan")))
                if state_value is not None and np.isfinite(payload["models"][baseline]["macro"].get(metric, float("nan"))) else None
                for baseline in baseline_names
            }
    return {
        "version": "state_cell_eval_reanalysis_summary.v1",
        "protocol": payload["protocol"],
        "metrics": metrics,
        "aggregate_by_method": aggregate,
        "aggregate_by_mode": mode_summary,
        "official_baselines": {name: payload["models"][name]["macro"] for name in baseline_names},
        "state_minus_baseline": comparisons,
        "old_metric_validity": "INVALID_FOR_STATE_COMPARISON",
        "metric_mismatch_diagnosis": "PENDING_FORMAL_COMPARISON",
        "historical_evaluation_version": "d2_state_test_evaluation.v1" if old_payload else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("infer", "metrics", "all"), default="all")
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
        output.write_text(json.dumps({"evaluation": result, "summary": summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "version": PROTOCOL_VERSION, "methods": sorted(result["models"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
