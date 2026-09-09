# STATE 评价指标审计与 D2 重分析协议

## 审计结论

历史 D2 评价不能直接解释为 STATE 官方评价。历史实现把全部测试细胞混合后计算一次跨基因 Pearson，把细胞×基因误差作为 MAE，并用“预测组之间是否存在非零距离”代替扰动区分分数；这些定义都不同于 STATE/Cell-Eval。

STATE 的测试评价按条件分别构建每个扰动的伪总体和 NTC 对照，主要报告：

1. 论文时期的 L1 扰动区分分数（PDS）；
2. Pearson Delta；
3. 差异表达基因识别 AUPRC；
4. 显著差异表达基因上的 log2FC Spearman；
5. DE Overlap@N；
6. 显著 DE 基因数量的效应大小 Spearman。

Cell-Eval 完整套件还包括 MAE/MSE、Delta-MAE/Delta-MSE、PDS-L2/PDS-cosine、多个 Overlap/Precision 阈值、方向一致率、召回率、ROC-AUC 和显著基因数量。MMD 是训练/验证损失，不是论文主测试指标。

## 版本冻结

- 论文兼容 PDS：绝对扰动效应、L1 距离、排除靶基因，分数为 `1-2*rank/T`；随机预测约为0。
- 当前官方 Cell-Eval：使用带符号扰动效应并返回 `1-rank/T`；本轮并行报告，不混合两个尺度。
- Pearson Delta 主结果使用论文时期和当前代码共同采用的带符号变化；论文文字中“绝对变化”的表述另作敏感性分析。
- 当前环境固定 `cell-eval==0.8.2`，完整配置按 STATE 当前入口跳过 `pearson_edistance` 和 `clustering_agreement`。

## 数据与基线不变量

- 六个冻结检查点、2,000基因顺序、扰动词表、划分和现有测试记录不变。
- 每个测试记录继续使用固定的32个目标细胞；每个 Rest/Stim8hr/Stim48hr 条件固定128个训练允许使用的 NTC 细胞作为共同评价对照。
- 模型预测、真实目标细胞和共同 NTC 均进入 Cell-Eval；原始 `.h5ad` 不修改。
- 条件均值和扰动均值只用训练部分拟合；不再按 MAE 选唯一基线。
- 官方低秩线性基线需要完整匹配的固定基因嵌入。D2 面板与官方 Replogle 嵌入不具备该条件，因此仅保留旧线性结果作为非官方诊断。

## 原因分解

重分析同时保留：

- A：历史指标与历史基线的已发布结果；
- B：官方指标与历史基线；
- C：官方指标与校正后的训练集均值基线。

只有当新版 PDS 或 Pearson Delta 的 STATE 优势在至少两个条件中方向一致、配对95%区间不跨0，并且在 C 中仍保持，才把“指标选择导致旧结论失真”标记为 `SUPPORTED`；否则分别标记 `PARTIAL` 或 `NOT_SUPPORTED`。

## 第一手来源

- [STATE v2 论文](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full.pdf)
- [STATE 预测入口](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/src/state/_cli/_tx/_predict.py#L983-L1014)
- [Cell-Eval 表达与 PDS 实现](https://github.com/ArcInstitute/cell-eval/blob/6928cf8bd7a706040ccfd13119e4085726dee64a/src/cell_eval/metrics/_anndata.py#L24-L223)
- [Cell-Eval 差异表达实现](https://github.com/ArcInstitute/cell-eval/blob/6928cf8bd7a706040ccfd13119e4085726dee64a/src/cell_eval/metrics/_de.py#L11-L269)
- [论文时期 PDS 实现](https://github.com/ArcInstitute/cell-eval/blob/162997c1bd0e6b52e7abac6a9b4939c1d394a01e/src/cell_eval/metrics/_anndata.py#L129-L198)
- [STATE 官方复现仓库](https://github.com/ArcInstitute/State-reproduce/tree/1a29ba1aa57ae996a57f77ec273e8547e824835b)
