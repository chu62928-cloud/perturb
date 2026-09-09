# D2 STATE 官方指标重分析报告

本报告只重新推理和评价六个已冻结检查点；没有重新训练，也没有修改面板、扰动词表、划分或历史评价文件。

- 协议：`state_cell_eval_reanalysis.v1`；Cell-Eval：`0.8.2`
- 测试记录：1793 个组合；固定测试流种子：`20260909`
- 基因顺序哈希：`810bed53174adcb51c80b39f00f28a4ac9031ae0e0b1ef7a47a4f7c33b7b1e99`
- 扰动词表哈希：`5c2c29e7f512f0cdf9fdf2ef31a3afeee8b286c76385f622bfe5e6f9065e7cc5`
- 划分哈希：`6debd4dbad36bccc22e47041a1c8461710ecfe5d715d0599740be4f8b809e84b`
- 固定对照：每条件 128 个训练部分 NTC 细胞；每个测试组合 32 个测试输入细胞

## 评价口径

论文主指标按扰动形成伪总体后，在条件内对扰动宏平均，再对三个条件等权平均。PDS 使用绝对扰动效应、L1 距离并排除靶基因，报告 `1−2×rank/T`；Pearson Delta 主分析使用带符号的扰动−NTC变化。Cell-Eval 0.8.2 完整配置同时保留差异表达、误差、重叠和当前版 PDS 指标；按官方预测入口跳过 `pearson_edistance` 与 `clustering_agreement`。

## 三种子结果

| 方法 | PDS-L1（均值） | Pearson Delta（均值） | PR-AUC（均值） | DE LFC Spearman（均值） |
|---|---:|---:|---:|---:|
| Scratch | 0.4183 [0.3790, 0.4609] | 0.5377 [0.5246, 0.5508] | 0.2262 [0.2021, 0.2469] | 0.5760 [0.5456, 0.6027] |
| Transfer | 0.2550 [0.2148, 0.2958] | 0.4677 [0.4432, 0.4887] | 0.0992 [0.0689, 0.1212] | 0.4071 [0.3689, 0.4446] |
| condition_mean（校正） | 0.0024 | 0.3419 | 0.0045 | 0.5003 |
| perturbation_mean（校正） | 0.2315 | 0.1740 | 0.0163 | 0.5101 |

区间是固定三个种子的分层自助法区间：先在扰动层做条件内宏平均，再按种子和条件重采样；它不是供者泛化区间。每个条件的有限扰动数和全部 Cell-Eval 指标保存在机器可读 JSON 及远端缓存中。

## A/B/C 原因分解

- **A：历史自定义指标＋历史基线。** 历史全局 Pearson、非零距离区分指标和细胞级 MAE 仅作旧结果对照，不能解释为 STATE 论文指标。
- **B：官方指标＋历史基线。** `old_condition_mean` 和 `old_perturbation_mean` 按旧版 train+validation 拟合规则单独重建，和新版指标并行计算。
- **C：官方指标＋校正基线。** `condition_mean` 和 `perturbation_mean` 只使用训练部分，作为最终公平比较。
- **结果：** Scratch 的 PDS-L1/Pearson Delta/PR-AUC 均值为 0.4183/0.5377/0.2262，Transfer 为 0.2550/0.4677/0.0992；两者相对校正基线在主要扰动特异性指标上均满足 `SUPPORTED`，但条件均值的总体表达误差仍较低。
- **原因判定：** 历史指标下条件均值领先，而官方指标与校正基线下两种 STATE 模式均稳定领先，因此本轮将“指标选择导致旧结论失真”标记为 `SUPPORTED`；这不等同于模型已通过真实细胞状态有效性验证。

历史指标有效性固定为：`INVALID_FOR_STATE_COMPARISON`。

| 模式 | 结论 | 解释摘要 |
|---|---|---|
| Scratch | **SUPPORTED** | 历史基线领先=True；PDS/Pearson 经校正后稳定支持=True；差异表达指标支持=True |
| Transfer | **SUPPORTED** | 历史基线领先=True；PDS/Pearson 经校正后稳定支持=True；差异表达指标支持=True |

本次新版主扰动特异性指标已稳定超过校正基线，因此按协议判定为 `SUPPORTED`；任何后续训练或检查点变更都必须另建协议版本，不能依据本次测试集结果回改。`MODEL_STATE_VALID` 仍保持 `NOT_EVALUABLE`。

## 复现

```powershell
$env:PYTHONPATH = 'src'
python scripts/evaluate_state_d2_cell_eval.py --phase infer ...
python scripts/evaluate_state_d2_cell_eval.py --phase metrics ... --report-md research/state_d2/D2_STATE_CELL_EVAL_REANALYSIS.md
python scripts/evaluate_state_d2_cell_eval.py --phase summarize ... --report-md research/state_d2/D2_STATE_CELL_EVAL_REANALYSIS.md
```

历史文件 `research/state_d2/d2_state_test_evaluation.json` 和 `D2_STATE_FINAL_REPORT.md` 未覆盖。
