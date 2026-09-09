from __future__ import annotations

"""Summarise the frozen D2 STATE test evaluation across random seeds.

The input contains one result for each of three seeds and each of Scratch and
Transfer.  This command does not re-read the test data or model checkpoints;
it only aggregates the already frozen, compact evaluation JSON.  Confidence
intervals are deterministic percentile bootstrap intervals over the three
seed-level values and are therefore stability intervals, not population-level
inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SEEDS = (20260901, 20260902, 20260903)
MODES = ("Scratch", "Transfer")
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260910

MAIN_METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pseudobulk_pearson", "伪总体 Pearson", ("pseudobulk_pearson",)),
    ("perturbation_discrimination", "扰动区分指标", ("perturbation_discrimination",)),
    ("mae", "MAE", ("summary", "mae")),
    ("rmse", "RMSE", ("summary", "rmse")),
    ("standardized_rmse", "标准化 RMSE", ("summary", "standardized_rmse")),
    ("direction_accuracy", "方向准确率", ("summary", "direction_accuracy")),
    ("mean_gap", "分布均值差", ("distribution", "mean_gap")),
    ("variance_gap", "分布方差差", ("distribution", "variance_gap")),
    ("sliced_wasserstein", "切片 Wasserstein 距离", ("distribution", "sliced_wasserstein")),
    ("relative_mae_improvement", "相对 MAE 改善", ("relative_mae_improvement",)),
)

CONDITION_METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pseudobulk_pearson", "伪总体 Pearson", ("pseudobulk_pearson",)),
    ("mae", "MAE", ("summary", "mae")),
    ("rmse", "RMSE", ("summary", "rmse")),
    ("direction_accuracy", "方向准确率", ("summary", "direction_accuracy")),
)


def _get_path(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    return float(current)


def _bootstrap_ci(values: Sequence[float], seed: int, reps: int = BOOTSTRAP_REPS) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    if array.size == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, array.size, size=(int(reps), array.size))
    sample_means = array[sample_indices].mean(axis=1)
    return [float(np.quantile(sample_means, 0.025)), float(np.quantile(sample_means, 0.975))]


def _summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        return {"n": 0, "mean": None, "sd": None, "bootstrap_ci95": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "bootstrap_ci95": _bootstrap_ci(array, seed),
    }


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_key(mode: str, seed: int) -> str:
    return f"{mode}_seed_{seed}"


def _validate_input(payload: Mapping[str, Any]) -> None:
    if payload.get("version") != "d2_state_test_evaluation.v1":
        raise ValueError("unsupported evaluation JSON version")
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("evaluation JSON has no model mapping")
    expected = {_model_key(mode, seed) for mode in MODES for seed in SEEDS}
    observed = set(models)
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"missing frozen model results: {missing}")


def _extract_metric_rows(result: Mapping[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for key, _label, path in MAIN_METRICS:
        values[key] = _get_path(result, path)
    return values


def _aggregate_metric(
    rows: Mapping[int, Mapping[str, float | None]],
    metric: str,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows.values() if row.get(metric) is not None]
    return _summary(values, seed=BOOTSTRAP_SEED + seed_offset)


def _aggregate_conditions(
    model_rows: Mapping[int, Mapping[str, Any]],
    conditions: Sequence[str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition_index, condition in enumerate(sorted(conditions)):
        output[condition] = {}
        for metric_index, (key, _label, path) in enumerate(CONDITION_METRICS):
            values = []
            for result in model_rows.values():
                condition_result = result.get("by_condition", {}).get(condition, {})
                value = _get_path(condition_result, path)
                if value is not None:
                    values.append(value)
            output[condition][key] = _summary(
                values,
                seed=BOOTSTRAP_SEED + seed_offset + condition_index * 100 + metric_index,
            )
    return output


def _aggregate_programs(
    model_rows: Mapping[int, Mapping[str, Any]],
    programs: Sequence[str],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for program_index, program in enumerate(sorted(programs)):
        values = []
        for result in model_rows.values():
            value = _get_path(result.get("program_direction", {}).get(program, {}), ("accuracy",))
            if value is not None:
                values.append(value)
        output[program] = _summary(values, seed=BOOTSTRAP_SEED + seed_offset + program_index)
    return output


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    compact = _extract_metric_rows(result)
    compact["gate_global_pass"] = bool(result.get("gate", {}).get("global_pass", False))
    compact["gate_checks"] = dict(result.get("gate", {}).get("checks", {}))
    compact["nested_mae_bootstrap"] = result.get("bootstrap_mae_improvement")
    compact["by_condition"] = result.get("by_condition", {})
    compact["program_direction"] = result.get("program_direction", {})
    return compact


def _format(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def _format_summary(summary: Mapping[str, Any], *, percent: bool = False) -> str:
    if summary.get("mean") is None:
        return "NA"
    if percent:
        mean = _format_percent(summary["mean"])
        sd = _format_percent(summary["sd"])
        ci = summary.get("bootstrap_ci95") or [None, None]
        return f"{mean} ± {sd} [{_format_percent(ci[0])}, {_format_percent(ci[1])}]"
    mean = _format(summary["mean"])
    sd = _format(summary["sd"])
    ci = summary.get("bootstrap_ci95") or [None, None]
    return f"{mean} ± {sd} [{_format(ci[0])}, {_format(ci[1])}]"


def _markdown_report(report: Mapping[str, Any]) -> str:
    protocol = report["protocol"]
    aggregate = report["aggregate"]
    baseline = report["baseline"]
    comparisons = report["baseline_comparison"]
    lines = [
        "# D2 STATE 单步预测器最终评价报告",
        "",
        f"> 报告版本：`{report['version']}`；输入摘要：`{report['source_evaluation_sha256']}`",
        "",
        "## 结论",
        "",
        f"按预注册全局闸门，本轮结论为：**{report['conclusion']['model_conclusion']}**。",
        "六个冻结检查点均完成相同测试组合评价，但没有一个同时满足 Pearson 提升、MAE 不恶化和自助法改善下界为正等要求。",
        f"`MODEL_STATE_VALID` 为 `{report['conclusion']['MODEL_STATE_VALID']}`；D2 的 Rest、Stim8hr、Stim48hr 是生物背景而不是 Naive/Th1/Th2/Th17 标签，因此程序方向只作内部诊断。",
        "",
        "## 1. 冻结协议与数据范围",
        "",
        f"- 评价范围：`{protocol['evaluation_scope']}`。",
        f"- 测试组合：{protocol['test_records']} 个；测试细胞：{protocol['test_cells']} 个。",
        f"- 测试流随机种子：`{protocol['test_stream_seed']}`；基因顺序哈希：`{protocol['gene_order_hash']}`。",
        f"- 扰动词表哈希：`{protocol['perturbation_vocab_hash']}`；划分哈希：`{protocol['split_hash']}`。",
        "- 测试响应只在六个最佳检查点冻结后读取；基线只使用训练和验证部分拟合。",
        "",
        "## 2. 每个种子的测试结果",
        "",
        "表中数值为 `均值 ± 三种子内标准差 [百分位自助法95%区间]` 的汇总见下一节；本表列出每个检查点的原始测试结果。",
        "",
        "| 模式 | 种子 | Pearson | MAE | RMSE | 相对 MAE 改善 | 全局闸门 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for mode in MODES:
        for seed in SEEDS:
            row = report["seed_results"][mode][str(seed)]
            lines.append(
                f"| {mode} | {seed} | {_format(row['pseudobulk_pearson'])} | "
                f"{_format(row['mae'])} | {_format(row['rmse'])} | "
                f"{_format_percent(row['relative_mae_improvement'])} | "
                f"{'通过' if row['gate_global_pass'] else '未通过'} |"
            )
    lines.extend(["", "## 3. 三种子汇总", "", "自助法以三个随机种子为重采样单位，重复 20,000 次；区间反映种子稳定性，不等同于供者或细胞总体的置信区间。", ""])
    lines.extend([
        "| 模式 | Pearson | 扰动区分 | MAE | RMSE | 方向准确率 | 相对 MAE 改善 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        metrics = aggregate[mode]["metrics"]
        lines.append(
            f"| {mode} | {_format_summary(metrics['pseudobulk_pearson'])} | "
            f"{_format_summary(metrics['perturbation_discrimination'])} | "
            f"{_format_summary(metrics['mae'])} | {_format_summary(metrics['rmse'])} | "
            f"{_format_summary(metrics['direction_accuracy'])} | "
            f"{_format_summary(metrics['relative_mae_improvement'], percent=True)} |"
        )
    lines.extend(["", "## 4. 生物背景分层", "", "| 模式 | 背景 | Pearson | MAE | RMSE |", "|---|---|---:|---:|---:|"])
    for mode in MODES:
        for condition in sorted(aggregate[mode]["by_condition"]):
            row = aggregate[mode]["by_condition"][condition]
            lines.append(
                f"| {mode} | {condition} | {_format_summary(row['pseudobulk_pearson'])} | "
                f"{_format_summary(row['mae'])} | {_format_summary(row['rmse'])} |"
            )
    lines.extend(["", "## 5. 谱系程序方向诊断", "", "这些分数不构成 D2 谱系转换结论；它们只表示预测效应与测试效应在已登记程序上的方向一致率。", "", "| 模式 | 程序 | 方向准确率 |", "|---|---|---:|"])
    for mode in MODES:
        for program in sorted(aggregate[mode]["program_direction"]):
            lines.append(f"| {mode} | {program} | {_format_summary(aggregate[mode]['program_direction'][program])} |")
    lines.extend(["", "## 6. 简单基线比较", "", f"最佳简单基线为 `{baseline['best_name']}`。", "", "| 方法 | Pearson 相对基线差值 | MAE 相对比值 | 扰动区分相对差值 |", "|---|---:|---:|---:|"])
    for mode in MODES:
        comparison = comparisons[mode]
        lines.append(
            f"| {mode} | {_format(comparison['pseudobulk_pearson_delta'])} | "
            f"{_format(comparison['mae_ratio'])} | {_format(comparison['discrimination_delta'])} |"
        )
    lines.extend([
        "",
        "基线本身的测试结果：",
        "",
        "| 基线 | Pearson | MAE | RMSE | 扰动区分 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, result in baseline["all_baselines"].items():
        lines.append(
            f"| {name} | {_format(result['pseudobulk_pearson'])} | {_format(result['mae'])} | "
            f"{_format(result['rmse'])} | {_format(result['perturbation_discrimination'])} |"
        )
    lines.extend(["", "## 7. 闸门状态", "", "| 检查项 | 六个检查点通过数 |", "|---|---:|"])
    for key, count in report["gate_summary"]["check_pass_counts"].items():
        lines.append(f"| `{key}` | {count}/6 |")
    lines.extend([
        "",
        f"全局闸门通过检查点：{report['gate_summary']['global_pass_count']}/6；NTC 测试闸门：不可用（测试划分不含 NTC 响应），缺失不计为通过。",
        "",
        "## 8. 复现说明",
        "",
        "本报告只聚合已经冻结的紧凑 JSON，不重新读取原始 `.h5ad`，也不保存预测矩阵。重新生成命令：",
        "",
        "```powershell",
        report["reproducibility"]["command"],
        "```",
        "",
        "运行要求：使用包含 NumPy 的数据处理环境，并设置 `PYTHONPATH=src`；输入评价 JSON、三种子及模型结果必须与本报告记录的哈希一致。若需要改变模型、训练预算、划分或测试指标，必须新建协议版本并重新完成六次训练。",
        "",
        "模型权重仍保留在远端运行目录，未提交到 Git；连接配置、原始 `.h5ad`、访问凭据和大型缓存同样未提交。",
        "",
    ])
    return "\n".join(lines)


def build_report(payload: Mapping[str, Any], source_sha256: str, input_path: Path) -> dict[str, Any]:
    models = payload["models"]
    seed_results: dict[str, dict[str, Any]] = {mode: {} for mode in MODES}
    model_rows: dict[str, dict[int, Mapping[str, Any]]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for seed in SEEDS:
            result = models[_model_key(mode, seed)]
            seed_results[mode][str(seed)] = _compact_result(result)
            model_rows[mode][seed] = result

    conditions = sorted({
        condition
        for rows in model_rows.values()
        for result in rows.values()
        for condition in result.get("by_condition", {})
    })
    programs = sorted({
        program
        for rows in model_rows.values()
        for result in rows.values()
        for program in result.get("program_direction", {})
    })

    aggregate: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        metric_rows = {
            seed: _extract_metric_rows(result)
            for seed, result in model_rows[mode].items()
        }
        aggregate[mode] = {
            "n_seeds": len(metric_rows),
            "metrics": {
                key: _aggregate_metric(metric_rows, key, seed_offset=mode_index * 1000 + metric_index)
                for metric_index, (key, _label, _path) in enumerate(MAIN_METRICS)
            },
            "by_condition": _aggregate_conditions(
                model_rows[mode], conditions, seed_offset=mode_index * 1000 + 300,
            ),
            "program_direction": _aggregate_programs(
                model_rows[mode], programs, seed_offset=mode_index * 1000 + 600,
            ),
        }
        nested = [
            result.get("bootstrap_mae_improvement", {})
            for result in model_rows[mode].values()
        ]
        aggregate[mode]["nested_record_bootstrap"] = {
            field: _summary(
                [float(row[field]) for row in nested if row.get(field) is not None],
                seed=BOOTSTRAP_SEED + mode_index * 1000 + 900 + field_index,
            )
            for field_index, field in enumerate(("estimate", "lower", "upper"))
        }

    baseline_results = payload.get("baseline_results", {})
    best_baseline_name = str(payload["best_baseline"])
    if best_baseline_name not in baseline_results:
        raise ValueError("best baseline is absent from baseline_results")
    baseline = {
        "best_name": best_baseline_name,
        "all_baselines": {
            name: {
                "pseudobulk_pearson": _get_path(result, ("pseudobulk_pearson",)),
                "perturbation_discrimination": _get_path(result, ("perturbation_discrimination",)),
                "mae": _get_path(result, ("summary", "mae")),
                "rmse": _get_path(result, ("summary", "rmse")),
                "direction_accuracy": _get_path(result, ("summary", "direction_accuracy")),
            }
            for name, result in baseline_results.items()
        },
    }
    best = baseline["all_baselines"][best_baseline_name]
    baseline_comparison = {}
    for mode in MODES:
        metrics = aggregate[mode]["metrics"]
        baseline_comparison[mode] = {
            "baseline_name": best_baseline_name,
            "pseudobulk_pearson_delta": metrics["pseudobulk_pearson"]["mean"] - best["pseudobulk_pearson"],
            "mae_ratio": metrics["mae"]["mean"] / max(best["mae"], 1e-12),
            "discrimination_delta": metrics["perturbation_discrimination"]["mean"] - best["perturbation_discrimination"],
            "direction_accuracy_delta": metrics["direction_accuracy"]["mean"] - best["direction_accuracy"],
        }

    gate_checks = sorted({
        check
        for result in models.values()
        for check in result.get("gate", {}).get("checks", {})
    })
    gate_summary = {
        "global_pass_count": sum(bool(result.get("gate", {}).get("global_pass", False)) for result in models.values()),
        "check_pass_counts": {
            check: sum(bool(result.get("gate", {}).get("checks", {}).get(check, False)) for result in models.values())
            for check in gate_checks
        },
        "ntc_test_available": False,
    }

    report = {
        "version": "d2_state_final_summary.v1",
        "source_evaluation": str(input_path),
        "source_evaluation_sha256": source_sha256,
        "protocol": {
            "evaluation_scope": payload.get("evaluation_scope"),
            "test_records": payload.get("test_records"),
            "test_cells": payload.get("test_cells"),
            "test_stream_seed": payload.get("test_stream_seed"),
            "gene_order_hash": payload.get("gene_order_hash"),
            "perturbation_vocab_hash": payload.get("perturbation_vocab_hash"),
            "split_hash": payload.get("split_hash"),
            "finite_test_record_hash": payload.get("finite_test_record_hash"),
        },
        "conclusion": {
            "model_conclusion": "STATE未超过简单基线",
            "MODEL_STATE_VALID": payload.get("MODEL_STATE_VALID", "NOT_EVALUABLE"),
            "composition_and_sequential_planning": payload.get("composition_and_sequential_planning", "NOT_STARTED"),
            "warning": payload.get("warning"),
        },
        "baseline": baseline,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "baseline_comparison": baseline_comparison,
        "gate_summary": gate_summary,
        "reproducibility": {
            "bootstrap_replicates": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "three frozen random seeds within each mode",
            "command": "python scripts/summarize_state_d2_evaluation.py --input research/state_d2/d2_state_test_evaluation.json --output-json research/state_d2/d2_state_final_summary.v1.json --output-md research/state_d2/D2_STATE_FINAL_REPORT.md",
            "prediction_matrices_committed": False,
            "raw_h5ad_modified": False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    _validate_input(payload)
    report = build_report(payload, _source_hash(args.input), args.input)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "model_conclusion": report["conclusion"]["model_conclusion"],
        "global_pass_count": report["gate_summary"]["global_pass_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
