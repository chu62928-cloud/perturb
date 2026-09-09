# CD4 T 细胞扰动预测项目

这是一个面向 CD4 T 细胞 Perturb-seq 的可审计分析与建模仓库。项目把原始数据审计、基因面板冻结、扰动划分、STATE 单步预测、基线评价和后续组合规划分开管理；原始 `.h5ad` 与模型权重不进入 Git。

## 当前阶段：D2 STATE 单步预测

当前唯一大目标分支为 `d2-state-single-step`。D2 使用 Rest、Stim8hr、Stim48hr 作为生物背景，按“扰动基因 × 生物背景”留出测试组合：训练、验证和测试分别冻结，测试响应直到六个最佳检查点全部完成后才读取。

三种随机种子下的 Scratch 与 Transfer 共六个检查点已经完成训练和封存测试评价。测试集包含 1,793 个组合、57,376 个细胞。最终按预注册闸门得到的结论是：**STATE 未超过最佳简单基线 `condition_mean`**。这表示本协议下的总体表达预测没有达到预设优势，不代表可以据此宣称或否定谱系转换；`MODEL_STATE_VALID` 仍为 `NOT_EVALUABLE`。

最终报告与机器可读汇总：

- [D2 STATE 最终评价报告](research/state_d2/D2_STATE_FINAL_REPORT.md)
- [D2 STATE 汇总 JSON](research/state_d2/d2_state_final_summary.v1.json)
- [封存测试评价 JSON](research/state_d2/d2_state_test_evaluation.json)
- [完整流水线说明](README_PIPELINE.md)

## 重新生成汇总报告

汇总脚本只读取已经冻结的紧凑评价 JSON，不重新读取原始数据，也不保存预测矩阵。建议在数据处理环境中运行：

```powershell
$env:PYTHONPATH = "src"
python scripts/summarize_state_d2_evaluation.py `
  --input research/state_d2/d2_state_test_evaluation.json `
  --output-json research/state_d2/d2_state_final_summary.v1.json `
  --output-md research/state_d2/D2_STATE_FINAL_REPORT.md
```

报告中的三种子区间采用固定随机种子和 20,000 次百分位自助法重采样；重采样单位是三个冻结随机种子，因此应理解为种子稳定性区间，而不是供者或细胞总体的置信区间。

## 环境与测试

- `e3_pipeline`：数据审计、HVG、划分、评价汇总和普通测试；
- `e3_state`：PyTorch、CUDA 和 STATE 训练/推理；
- 运行测试前设置 `PYTHONPATH=src`，远端数据处理环境最近一次验证为 `41 passed, 5 skipped`（跳过项是该环境未安装 PyTorch 的深度测试）。

每次提交前执行：

```powershell
git status --short
python -m pytest
git diff --check
```

Git 提交只保留代码、审计摘要、冻结划分和紧凑指标；不得提交连接配置、令牌、原始 `.h5ad`、检查点、大型缓存或生成矩阵。
