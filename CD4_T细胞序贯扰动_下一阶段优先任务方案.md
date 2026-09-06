# CD4 T 细胞序贯扰动项目：下一阶段优先任务方案

**版本日期：2026-09-02**

## 1. 当前阶段判断

当前项目已经完成主要工程骨架，但尚未进入"序贯规划有效"的科学验证阶段。

已经完成或基本完成：

-   Snakemake 主流程 `preflight → audit → prepare_pilot → freeze_data`
    已实际跑通并可重跑。
-   96 基因试点已拆分为 `primary_64`（响应盲、主要统计评价）与
    `challenge_32`（压力测试），两者无重叠。
-   `data_contract_frozen_v1.json` 已生成，目前状态为
    `DATA_CONTRACT_READY`，但尚未升级为 `DATA_VALID`。
-   D2 响应尚未用于开发，继续保持锁定。
-   连续状态接口、三类留出、D1 分层 Bootstrap、`MODEL_GLOBAL_VALID` /
    `MODEL_STATE_VALID`
    双闸门、组合性几何匹配封存、多目标安全接口等工程框架已实现。
-   基线、支持度、组合性、模型分歧、锁定发布、束搜索等非深度模块已经完成
    smoke test。

当前尚未成立：

`DATA_VALID → MODEL_GLOBAL_VALID → MODEL_STATE_VALID → COMPOSITION_VALID → PLANNER_VALID → EXPERIMENT_VALID`

因此下一阶段的目标不是立即做 Th2→Th17，也不是扩展
Planner，而是完成可靠的 D1 单步
benchmark，并用一个更简单、可验证的免疫分化过程（优先考虑 naive CD4 →
Th1）建立第一版 fate/trajectory 验证框架。

------------------------------------------------------------------------

## 2. 下一阶段总目标

### Milestone A：DATA_VALID + D1 Benchmark Frozen

完成数据、guide、特征空间、状态空间和评测划分冻结，使所有后续模型在同一套不可随意修改的
benchmark 上比较。

### Milestone B：证明或否定可靠的单步状态转移

依次比较简单基线、`pert2state`、STATE zero-shot，以及必要时的 STATE
轻量适配，回答：

1.  给定基因 KD，能否预测总体单步扰动效应？
2.  给定起始局部细胞状态和基因 KD，能否预测状态依赖的单步扰动效应？

只有这两个问题均通过，才进入第二执行器和两步组合性。

### Milestone C：建立简单分化任务的 fate evaluator

第一版优先采用 naive CD4 → Th1
或其他具有清晰参考轨迹的免疫分化过程，而不是直接挑战 Th2→Th17。

核心目标不是手工定义一个"Th1 分数"，而是建立：

`预测扰动状态 → reference trajectory / fate landscape → 分化进展与方向评价`

CellRank 或 CellRank-like 的 transition/fate
思想作为候选方案；stress、apoptosis、cell cycle、generic activation 等
gene programs 仅作为辅助诊断和安全约束，不作为主要 fate 定义。

------------------------------------------------------------------------

# 3. 优先级 P0：立即完成

## P0-1. 将数据契约从 DATA_CONTRACT_READY 升级为 DATA_VALID

### 必须完成

-   完成 guide_id → target_gene_id / target_gene_name 校订。
-   输出 guide mapping rate 与异常映射清单。
-   冻结 normalization 配置与 hash。
-   冻结共享 gene order 与 hash。
-   冻结 primary/challenge 集合及其 hash。
-   冻结 gene/state/gene×state split，并保存 split hash。
-   明确记录 `d2_responses_used=false`。
-   原始 `.h5ad` 保持只读。

### 验收标准

只有所有关键数据契约字段、hash 和 QC 证据齐全后，才声明：

`DATA_VALID = TRUE`

在此之前不进入正式模型比较。

------------------------------------------------------------------------

## P0-2. 构建 D1 单步扰动 ground truth

这是当前最重要的数据分析任务。

### 计算原则

按 condition 匹配 NTC，首先保留 guide-level 效应：

`Effect(guide, condition) = mean(KD guide cells) - mean(condition-matched NTC)`

随后以稳健方式聚合为 gene-level effect。

### 同时输出

-   每个 gene 的有效 guide 数。
-   每条 guide 的有效细胞数。
-   guide 间一致性。
-   gene-level effect size。
-   condition dependence / stimulation dependence。
-   不稳定或冲突 guide 标记。
-   primary_64 与 challenge_32 的完整 QC 表。

### 注意

`primary_64` 的主要泛化统计不能因为观察 D1 响应后重新选择成员。

`challenge_32` 可以使用响应信息做压力测试，但不能进入主要模型放行统计。

------------------------------------------------------------------------

## P0-3. 建立真实连续状态空间与严格留出

基于 D1 NTC 建立固定的连续潜空间。

### 建议保持当前方案

-   固定 50 维潜空间。
-   使用局部近邻区域作为主要 state 单位。
-   Leiden 仅用于可视化、描述和诊断。
-   保留局部邻域、缓冲区和支持度信息。

### 正式生成三类 split

1.  Gene holdout
2.  State-region holdout
3.  Gene × state joint holdout

第三类是最重要的 Planner 前置 benchmark。

### 输出

-   latent embedding
-   neighborhood definition
-   train/validation/test membership
-   buffer regions
-   local cell density
-   perturbation support
-   split hash

------------------------------------------------------------------------

# 4. 优先级 P1：完成第一个科学 Benchmark

## P1-1. 先跑简单基线

建议顺序：

1.  No-change
2.  Condition mean
3.  Gene-effect transfer
4.  Ridge / linear residual model
5.  pert2state

不要先假设 foundation model 一定优于简单模型。

需要回答的是：

> STATE 是否提供了超出简单低阶模型的可重复增益？

如果线性模型与 STATE 相当，这本身就是重要结果，应据此调整 executor
设计。

------------------------------------------------------------------------

## P1-2. 运行 STATE zero-shot

第一轮 STATE 不做 fine-tuning。

统一使用与 baseline 完全相同的：

-   gene split
-   state split
-   gene×state split
-   feature order
-   evaluation metrics
-   random seeds

### 主要评价拆成两个闸门

#### MODEL_GLOBAL_VALID

评价：

-   总体表达效应误差
-   扰动方向一致性
-   条件级稳定性
-   NTC 零效应误差
-   关键 readout 的合理性

#### MODEL_STATE_VALID

评价：

-   gene×state 联合留出
-   局部扰动方向
-   状态比例变化
-   分布距离
-   方差/异质性恢复
-   不同起始状态的稳定性

只有：

`MODEL_GLOBAL_VALID AND MODEL_STATE_VALID`

同时成立，才允许进入两步组合性。

------------------------------------------------------------------------

## P1-3. 决定是否需要 STATE adaptation

只有 zero-shot 结果显示 STATE 有潜力但未达到闸门时，才比较轻量适配：

-   output layer only
-   last layers
-   LoRA / low-rank adaptation

全部选择仅允许在 D1 内完成。

D2 不参与 adaptation strategy、超参数或阈值选择。

------------------------------------------------------------------------

# 5. 优先级 P1.5：建立 naive → Th1 简单分化验证任务

这一任务与模型单步 benchmark 可以部分并行，但不应阻塞 P0。

## 目标

先证明模型预测的扰动状态是否沿一个已知、较清晰的免疫分化方向移动，再挑战
Th2→Th17。

第一候选：

`naive CD4 → Th1`

如果现有参考数据不支持该轨迹，再选择数据证据更完整的简单免疫分化过程。

------------------------------------------------------------------------

## Fate evaluator 的建议结构

### 第一层：Perturbation executor

`S_t --KD(g)--> predicted S_(t+1)`

由 STATE / baseline / 第二执行器负责。

### 第二层：Trajectory / fate evaluator

评价 predicted state：

-   是否仍位于真实细胞 manifold 支持区域？
-   是否沿 reference differentiation trajectory 前进？
-   对目标 fate 的概率/吸收概率是否增加？
-   是否出现方向逆转或跳出合理轨迹？

CellRank 或 CellRank-like transition/fate framework 优先在这一层评估。

### 第三层：Safety / confounder diagnostics

保留少量 gene programs，例如：

-   stress
-   apoptosis
-   cell cycle / proliferation
-   generic activation
-   必要时 IFN response

这些程序主要用于排除"假分化进展"，而不是人工定义 Th1 fate。

------------------------------------------------------------------------

## 暂时不要做的事情

现阶段不要优先：

-   人工冻结完整 Th2/Th17 reward。
-   用单一 Th1/Th17 gene score 作为 Planner 的主要目标。
-   直接开展 Th2→Th17 两步搜索。
-   加入 ATAC 作为 Planner action probability。
-   加入 RNA velocity 作为 executor 输入。
-   引入强化学习。
-   引入 MCTS。
-   扩展到深度 3 以上。
-   大规模生成正式序贯候选。

RNA velocity / CellRank / ATAC 是否加入，应在简单 fate benchmark
建立后，根据"是否显著改善 fate direction 判断"再决定。

------------------------------------------------------------------------

# 6. 优先级 P2：只有单步双闸门通过后才启动

## 第二执行器

至少引入一个与 STATE 结构不同的 executor。

目的首先是：

-   独立验证
-   模型分歧
-   epistemic uncertainty

而不是简单做模型平均。

------------------------------------------------------------------------

## Proxy composition benchmark

至少满足当前预设：

-   ≥ 50 个有效 proxy triplets
-   ≥ 20 个第二步基因
-   ≥ 3 个局部状态区域

执行顺序必须是：

1.  根据第一步预测状态的几何关系匹配真实状态。
2.  检查 support / density / distance。
3.  冻结匹配清单。
4.  再读取第二步真实响应。
5.  评价第二次调用是否可靠。

通过后才声明：

`COMPOSITION_VALID`

------------------------------------------------------------------------

# 7. 优先级 P3：最后才进入 Planner

只有：

`DATA_VALID` → `MODEL_GLOBAL_VALID` → `MODEL_STATE_VALID` →
`COMPOSITION_VALID`

全部成立后，才正式启用 Planner。

第一版保持：

-   depth = 2
-   beam width = 32
-   不重复同一基因
-   仅 CRISPRi / KD
-   中间态 support gate
-   OOD gate
-   uncertainty / disagreement penalty
-   safety penalty

第一版 Planner 首先在 naive → Th1 等简单任务上验证。

只有简单任务证明：

> 两步序列能够比最佳单步扰动更稳定地推进目标 fate

之后，才升级到 Th2→Th17 等困难转换。

------------------------------------------------------------------------

# 8. 推荐的实际执行顺序

  顺序   任务                                              目标状态
  ------ ------------------------------------------------- ----------------------
  1      guide 校订 + QC                                   完成数据基础
  2      normalization / gene order / split hash 冻结      DATA_VALID
  3      D1 单步 effect matrix                             Ground truth ready
  4      连续 latent state + 三类 holdout                  Benchmark frozen
  5      简单 baseline                                     Baseline established
  6      STATE zero-shot                                   初步 executor 评价
  7      GLOBAL / STATE 双闸门                             单步科学结论
  8      必要时 STATE adaptation                           改善单步性能
  9      naive→Th1 reference trajectory / fate evaluator   简单生物学验证
  10     第二 executor                                     模型分歧与独立支持
  11     proxy composition                                 COMPOSITION_VALID
  12     D2 一次性 provisional test                        外部供者暂定验证
  13     depth-2 Planner                                   PLANNER_VALID
  14     简单分化顺序实验                                  EXPERIMENT_VALID
  15     Th2→Th17                                          困难任务扩展

------------------------------------------------------------------------

# 9. 最近一个里程碑

当前不要把"训练 STATE"定义成下一个里程碑。

建议下一个正式里程碑固定为：

## Milestone A：DATA_VALID + D1 Benchmark Frozen

完成标准：

-   guide mapping 完成并通过 QC；
-   D1 effect matrix 完成；
-   primary_64 / challenge_32 正式冻结；
-   normalization / gene order / split hash 完整；
-   continuous state regions 完成；
-   三类 holdout 完成；
-   D2 保持未读取；
-   benchmark 可完全复现。

随后立即进入：

## Milestone B：Baseline vs STATE 单步决胜

这一阶段结束时必须能够明确回答：

1.  简单模型能做到什么程度？
2.  STATE 是否真正优于简单模型？
3.  优势是否存在于 state-dependent prediction，而不只是总体平均表达？
4.  模型是否已经具备作为序贯状态转移 executor 的最低资格？

在这四个问题回答清楚之前，不应把主要计算资源投入到 Planner 或复杂
Th2→Th17 搜索。
