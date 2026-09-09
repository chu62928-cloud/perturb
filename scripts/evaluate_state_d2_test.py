from __future__ import annotations

"""One-shot D2 test evaluation for the six frozen STATE checkpoints.

The script deliberately materializes one deterministic finite test stream in
memory and reuses it for every checkpoint and baseline.  No test response is
used to choose a checkpoint, a model hyperparameter, or a baseline setting.
Only compact JSON summaries are retained by the repository; prediction
matrices remain in the ignored runtime directory.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from cd4perturb.metrics import hierarchical_bootstrap
from cd4perturb.state_d2_data import D2BatchStream
from cd4perturb.state_d2_evaluation import evaluate_d2_predictions
from cd4perturb.state_d2_training import (
    build_official_state_adapter,
    initialize_transfer_adapter,
)


SEEDS = (20260901, 20260902, 20260903)
MODES = ("Scratch", "Transfer")
EVAL_SEED = 20260909
BOOTSTRAP_REPS = 1000


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _flatten_batches(batches: list[Mapping]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("finite D2 stream is empty")
    expression = np.concatenate([batch["expression"].cpu().numpy() for batch in batches], axis=0)
    target = np.concatenate([batch["target"].cpu().numpy() for batch in batches], axis=0)
    names = np.asarray([name for batch in batches for name in batch["perturbation_names"]], dtype=str)
    conditions = np.asarray([condition for batch in batches for condition in batch["conditions"]], dtype=str)
    records = np.asarray([key for batch in batches for key in batch["record_keys"]], dtype=str)
    return {
        "expression": expression,
        "target": target,
        "names": np.repeat(names, expression.shape[1]),
        "conditions": np.repeat(conditions, expression.shape[1]),
        "records": np.repeat(records, expression.shape[1]),
        "record_names": names,
        "record_conditions": conditions,
        "record_keys": records,
    }


def _program_direction(real: np.ndarray, predicted: np.ndarray,
                       control: np.ndarray, gene_order: list[str],
                       programs: Mapping[str, list[str]]) -> dict:
    index = {gene: i for i, gene in enumerate(gene_order)}
    result = {}
    true_effect = real.mean(axis=1) - control.mean(axis=1)
    pred_effect = predicted.mean(axis=1) - control.mean(axis=1)
    for name, genes in programs.items():
        columns = [index[gene] for gene in genes if gene in index]
        if not columns:
            result[name] = {"n_genes": 0, "n_records": 0, "accuracy": None}
            continue
        true_score = true_effect[:, columns].mean(axis=1)
        pred_score = pred_effect[:, columns].mean(axis=1)
        valid = np.abs(true_score) > 1e-8
        accuracy = float(np.mean(np.sign(true_score[valid]) == np.sign(pred_score[valid]))) if valid.any() else None
        result[name] = {"n_genes": len(columns), "n_records": int(valid.sum()),
                        "accuracy": accuracy, "genes": [gene_order[i] for i in columns]}
    return result


def _record_mae(real: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if real.shape != predicted.shape or real.ndim != 3:
        raise ValueError("record MAE requires [record, cell, gene] arrays")
    return np.mean(np.abs(real - predicted), axis=(1, 2))


def _bootstrap_improvement(real: np.ndarray, predicted: np.ndarray, baseline: np.ndarray,
                           record_names: np.ndarray, record_conditions: np.ndarray,
                           seed: int) -> dict:
    model_mae = _record_mae(real, predicted)
    base_mae = _record_mae(real, baseline)
    improvement = 1.0 - model_mae / np.maximum(base_mae, 1e-12)
    return hierarchical_bootstrap(
        improvement,
        [record_names, record_conditions],
        statistic=np.mean,
        n_boot=BOOTSTRAP_REPS,
        seed=seed,
        scope="D2_test_gene_x_condition_cell_sets",
    )


def _accumulate_target_stats(stream: D2BatchStream, *, seed: int,
                             records: Iterable[tuple[str, str]] | None = None) -> dict:
    """Collect compact condition/gene means and diagonal linear moments."""
    n_genes = len(stream.panel)
    condition_sum: dict[str, np.ndarray] = {}
    condition_count: dict[str, int] = {}
    gene_sum: dict[str, np.ndarray] = {}
    gene_count: dict[str, int] = {}
    gene_condition_sum: dict[tuple[str, str], np.ndarray] = {}
    gene_condition_count: dict[tuple[str, str], int] = {}
    sx = np.zeros(n_genes, dtype=np.float64)
    sy = np.zeros(n_genes, dtype=np.float64)
    sxx = np.zeros(n_genes, dtype=np.float64)
    sxy = np.zeros(n_genes, dtype=np.float64)
    syy = np.zeros(n_genes, dtype=np.float64)
    n = 0
    for batch in stream.iter_records(seed=seed, records=records):
        x = batch["expression"].cpu().numpy().astype(np.float64, copy=False)
        y = batch["target"].cpu().numpy().astype(np.float64, copy=False)
        for index, (gene, condition) in enumerate(zip(batch["perturbation_names"], batch["conditions"])):
            xi, yi = x[index], y[index]
            flat_x, flat_y = xi.reshape(-1, n_genes), yi.reshape(-1, n_genes)
            condition_sum[condition] = condition_sum.get(condition, np.zeros(n_genes)) + flat_y.sum(axis=0)
            condition_count[condition] = condition_count.get(condition, 0) + len(flat_y)
            gene_sum[gene] = gene_sum.get(gene, np.zeros(n_genes)) + flat_y.sum(axis=0)
            gene_count[gene] = gene_count.get(gene, 0) + len(flat_y)
            key = (gene, condition)
            gene_condition_sum[key] = gene_condition_sum.get(key, np.zeros(n_genes)) + flat_y.sum(axis=0)
            gene_condition_count[key] = gene_condition_count.get(key, 0) + len(flat_y)
            sx += flat_x.sum(axis=0)
            sy += flat_y.sum(axis=0)
            sxx += np.square(flat_x).sum(axis=0)
            sxy += (flat_x * flat_y).sum(axis=0)
            syy += np.square(flat_y).sum(axis=0)
            n += len(flat_y)
    if n == 0:
        raise ValueError("baseline fit stream is empty")
    return {"condition_sum": condition_sum, "condition_count": condition_count,
            "gene_sum": gene_sum, "gene_count": gene_count,
            "gene_condition_sum": gene_condition_sum,
            "gene_condition_count": gene_condition_count,
            "sx": sx, "sy": sy, "sxx": sxx, "sxy": sxy, "syy": syy, "n": n}


def _merge_stats(left: dict, right: dict) -> dict:
    merged = {key: value for key, value in left.items() if key not in {
        "condition_sum", "condition_count", "gene_sum", "gene_count",
        "gene_condition_sum", "gene_condition_count", "sx", "sy", "sxx", "sxy", "syy", "n"}}
    for field in ("condition_sum", "gene_sum", "gene_condition_sum"):
        keys = set(left[field]) | set(right[field])
        template = next(iter((left[field] or right[field]).values()))
        merged[field] = {
            key: left[field].get(key, np.zeros_like(template)) +
            right[field].get(key, np.zeros_like(template))
            for key in keys
        }
    for field in ("condition_count", "gene_count", "gene_condition_count"):
        keys = set(left[field]) | set(right[field])
        merged[field] = {key: int(left[field].get(key, 0) + right[field].get(key, 0)) for key in keys}
    for field in ("sx", "sy", "sxx", "sxy", "syy"):
        merged[field] = left[field] + right[field]
    merged["n"] = int(left["n"] + right["n"])
    return merged


def _fit_linear_coefficients(stats: dict, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    n = float(stats["n"])
    denom = stats["sxx"] - stats["sx"] ** 2 / n + float(alpha)
    slope = (stats["sxy"] - stats["sx"] * stats["sy"] / n) / np.maximum(denom, 1e-12)
    intercept = stats["sy"] / n - slope * stats["sx"] / n
    return slope.astype(np.float32), intercept.astype(np.float32)


def _baseline_predictions(flat: dict, stats: dict, slope: np.ndarray, intercept: np.ndarray) -> dict[str, np.ndarray]:
    x = flat["expression"].reshape(-1, flat["expression"].shape[-1])
    n_records, set_len, n_genes = flat["expression"].shape
    condition_means = {condition: stats["condition_sum"][condition] / stats["condition_count"][condition]
                       for condition in stats["condition_sum"]}
    gene_means = {gene: stats["gene_sum"][gene] / stats["gene_count"][gene]
                  for gene in stats["gene_sum"]}
    condition_pred = np.vstack([condition_means[condition] for condition in flat["record_conditions"]])
    gene_pred = []
    for gene, condition in zip(flat["record_names"], flat["record_conditions"]):
        # The held-out gene/background pair is absent by construction.  A
        # gene mean fitted on its other backgrounds is therefore the intended
        # perturbation-mean transfer; condition mean is a defensive fallback.
        gene_pred.append(gene_means.get(gene, condition_means[condition]))
    gene_pred = np.vstack(gene_pred)
    low_rank = (x * slope[None, :] + intercept[None, :]).reshape(n_records, set_len, n_genes)
    return {
        "no_change": flat["expression"].copy(),
        "condition_mean": np.repeat(condition_pred[:, None, :], set_len, axis=1),
        "perturbation_mean": np.repeat(gene_pred[:, None, :], set_len, axis=1),
        "low_rank_linear_residual": low_rank,
    }


def _select_alpha(val_batches: list[Mapping], val_flat: dict, train_stats: dict) -> tuple[float, dict]:
    target = val_flat["target"].astype(np.float32, copy=False)
    x = val_flat["expression"].astype(np.float32, copy=False)
    candidates = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    scores = {}
    for alpha in candidates:
        slope, intercept = _fit_linear_coefficients(train_stats, alpha)
        prediction = x * slope[None, None, :] + intercept[None, None, :]
        scores[str(alpha)] = float(np.mean(np.abs(prediction - target)))
    selected = min(candidates, key=lambda alpha: scores[str(alpha)])
    return float(selected), {"candidates": scores, "selected": float(selected),
                             "selection_split": "validation", "fit_split": "train"}


def _load_programs(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: list(value) for key, value in payload.get("identity_programs", {}).items()}


def _evaluate_one(real: np.ndarray, predicted: np.ndarray, flat: dict,
                  baseline: np.ndarray, control: np.ndarray, gene_order: list[str],
                  programs: Mapping[str, list[str]], model_name: str, seed: int) -> dict:
    # Pairwise discrimination and sliced-Wasserstein are capped internally by
    # the metric implementation for the 1,793 held-out combinations.  Full
    # test cells are retained for the primary Pearson/MAE/MSE quantities.
    result = evaluate_d2_predictions(
        real.reshape(-1, real.shape[-1]), predicted.reshape(-1, predicted.shape[-1]),
        flat["names"], flat["conditions"],
        baseline=baseline.reshape(-1, baseline.shape[-1]),
    )
    result["model"] = model_name
    result["bootstrap_mae_improvement"] = _bootstrap_improvement(
        real, predicted, baseline, flat["record_names"], flat["record_conditions"], seed)
    result["program_direction"] = _program_direction(real, predicted, control, gene_order, programs)
    result["test_records"] = int(len(real))
    result["test_cells"] = int(real.shape[0] * real.shape[1])
    result["evaluation_scope"] = "D2_test_gene_x_background_holdout"
    return result


def _gate(result: dict, baseline_result: dict) -> dict:
    model_pearson = float(result["pseudobulk_pearson"])
    base_pearson = float(baseline_result["pseudobulk_pearson"])
    model_mae = float(result["summary"]["mae"])
    base_mae = float(baseline_result["summary"]["mae"])
    model_discrimination = float(result["perturbation_discrimination"])
    base_discrimination = float(baseline_result["perturbation_discrimination"])
    checks = {
        "pearson_absolute_improvement_at_least_0_05": model_pearson - base_pearson >= 0.05,
        "bootstrap_mae_improvement_lower_bound_positive": result["bootstrap_mae_improvement"]["lower"] > 0,
        "discrimination_drop_at_most_0_02": model_discrimination - base_discrimination >= -0.02,
        "mae_not_worse_than_10_percent": model_mae <= 1.10 * max(base_mae, 1e-12),
        "ntc_test_available": False,
    }
    return {"checks": checks, "global_pass": bool(all(checks.values())),
            "relative_best_baseline": baseline_result.get("model", "baseline"),
            "pearson_delta": model_pearson - base_pearson,
            "mae_ratio": model_mae / max(base_mae, 1e-12),
            "discrimination_delta": model_discrimination - base_discrimination,
            "ntc_error": None,
            "note": "测试划分不含NTC响应，因此NTC零效应闸门本轮记为不可用，不以缺失当作通过。"}


def evaluate(args: argparse.Namespace) -> dict:
    paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
    panel_payload = json.loads(Path(args.gene_panel).read_text(encoding="utf-8"))
    vocab_payload = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    panel = list(panel_payload["gene_order"])
    names = list(vocab_payload["perturbation_names"])
    if len(panel) != 2000 or names[0] != "NTC":
        raise ValueError("evaluation requires the frozen 2,000-gene panel and NTC-first vocabulary")
    test_stream = D2BatchStream(paths, panel, names, splits, split="test", batch_size=args.batch_size,
                                set_len=32, seed=EVAL_SEED, device=None,
                                cache_expression=True, cache_max_bytes=args.cache_max_bytes)
    test_batches = list(test_stream.iter_records(seed=EVAL_SEED))
    test_flat = _flatten_batches(test_batches)
    real = test_flat["target"].reshape(-1, 32, len(panel))
    control = test_flat["expression"].reshape(-1, 32, len(panel))
    if len(test_flat["record_names"]) != len(set(test_flat["record_keys"].tolist())):
        raise AssertionError("finite test stream contains duplicate record keys")

    # Baseline fitting is restricted to train and validation; test responses
    # are not touched while selecting the diagonal ridge strength.
    train_stream = D2BatchStream(paths, panel, names, splits, split="train", batch_size=args.batch_size,
                                 set_len=32, seed=20260901, device=None,
                                 cache_expression=True, cache_max_bytes=args.cache_max_bytes)
    val_stream = D2BatchStream(paths, panel, names, splits, split="validation", batch_size=args.batch_size,
                               set_len=32, seed=20260902, device=None,
                               cache_expression=True, cache_max_bytes=args.cache_max_bytes)
    train_stats = _accumulate_target_stats(train_stream, seed=20260901)
    val_batches = list(val_stream.iter_records(seed=20260902))
    val_flat = _flatten_batches(val_batches)
    alpha, alpha_report = _select_alpha(val_batches, val_flat, train_stats)
    combined_stats = _merge_stats(train_stats, _accumulate_target_stats(val_stream, seed=20260902))
    slope, intercept = _fit_linear_coefficients(combined_stats, alpha)
    baseline_arrays = _baseline_predictions(test_flat, combined_stats, slope, intercept)
    baseline_results = {}
    for name, prediction in baseline_arrays.items():
        baseline_results[name] = evaluate_d2_predictions(
            real.reshape(-1, len(panel)), prediction.reshape(-1, len(panel)),
            test_flat["names"], test_flat["conditions"],
        )
        baseline_results[name]["model"] = name
    best_baseline = min(baseline_results, key=lambda name: baseline_results[name]["summary"]["mae"])

    programs = _load_programs(Path(args.programs))
    model_results = {}
    for mode in MODES:
        for seed in SEEDS:
            model_key = f"{mode}_seed_{seed}"
            checkpoint = Path(args.training_root) / mode / f"seed_{seed}" / "best.ckpt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            import torch
            model, checkpoint_payload = build_official_state_adapter(
                args.official_checkpoint, len(panel), len(names), cell_set_len=32)
            if mode == "Transfer":
                report = json.loads(Path(args.transfer_report).read_text(encoding="utf-8"))
                initialize_transfer_adapter(model, checkpoint_payload, panel, names, expected_report=report)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            contract_payload = json.loads(Path(args.contracts).read_text(encoding="utf-8"))
            expected_contracts = [row for row in contract_payload.get("contracts", [])
                                 if row.get("mode") == mode and int(row.get("seed", -1)) == seed]
            if len(expected_contracts) != 1 or payload.get("contract_hash") != hashlib.sha256(
                    json.dumps(expected_contracts[0], sort_keys=True).encode()).hexdigest():
                # Training contracts are frozen before test access.  A
                # mismatch is fatal: a checkpoint from another architecture
                # must never be evaluated under the D2 test protocol.
                raise ValueError(f"checkpoint contract mismatch for {model_key}")
            model.load_state_dict(payload["state_dict"])
            device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
            model = model.to(device).eval()
            prediction_batches = []
            with torch.no_grad():
                for batch in test_batches:
                    expression = batch["expression"].to(device)
                    perturbation = batch["perturbation"].to(device)
                    prediction = model(expression, perturbation).detach().cpu().numpy().astype(np.float32)
                    prediction_batches.append(prediction)
            prediction = np.concatenate(prediction_batches, axis=0)
            model_results[model_key] = _evaluate_one(
                real, prediction, test_flat, baseline_arrays[best_baseline], control,
                panel, programs, model_key, seed)
            model_results[model_key]["checkpoint"] = str(checkpoint)
            model_results[model_key]["best_baseline"] = best_baseline
            model_results[model_key]["gate"] = _gate(
                model_results[model_key], baseline_results[best_baseline])
            del model, prediction_batches, prediction
            if device == "cuda":
                torch.cuda.empty_cache()
    test_stream.close()
    train_stream.close()
    val_stream.close()
    return {
        "version": "d2_state_test_evaluation.v1",
        "evaluation_scope": "D2_test_gene_x_background_holdout",
        "test_responses_used": True,
        "test_stream_seed": EVAL_SEED,
        "test_records": int(len(test_flat["record_names"])),
        "test_cells": int(real.shape[0] * real.shape[1]),
        "gene_order_hash": panel_payload.get("gene_order_hash"),
        "perturbation_vocab_hash": vocab_payload.get("vocab_hash"),
        "split_hash": splits.get("split_hash"),
        "finite_test_record_hash": _json_hash(test_flat["record_keys"].tolist()),
        "baseline_alpha_selection": alpha_report,
        "baseline_results": baseline_results,
        "best_baseline": best_baseline,
        "models": model_results,
        "MODEL_STATE_VALID": "NOT_EVALUABLE",
        "composition_and_sequential_planning": "NOT_STARTED",
        "warning": "D2条件是生物背景，不是Naive/Th1/Th2/Th17标签；程序方向仅作内部诊断，不能宣称完成谱系转换。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True)
    parser.add_argument("--gene-panel", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--programs", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--official-checkpoint", required=True)
    parser.add_argument("--transfer-report", required=True)
    parser.add_argument("--contracts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-max-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "test_records": result["test_records"],
                      "test_cells": result["test_cells"], "best_baseline": result["best_baseline"],
                      "models": sorted(result["models"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
