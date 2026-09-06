# D2 专属 STATE 单步预测器执行规划

更新日期：2026-09-06

适用分支：`d2-state-single-step`

固定 STATE 代码：`9bbfe78a434a55205e4de834e1ea99f85f7a3add`

数据范围：D2 Rest、Stim8hr、Stim48hr

## 1. 本轮目标和停止边界

本轮只完成以下目标：

1. 建立并独立验证 Naive、Th1、Th2、Th17 转录程序评分器；
2. 冻结 D2 的数据合同、扰动词表、划分和恰好2,000个基因的表达面板；
3. 公平比较 `STATE-D2-Scratch` 与 `STATE-D2-Transfer`；
4. 与简单基线比较，并给出通过、未通过或不可评价的明确结论。

本轮不执行双扰动、序贯组合、束搜索或 Th2→Th17 规划。D2 的三个条件是刺激时间背景，不是 Naive、Th1、Th2、Th17 真值标签。外部极化数据用于验证评分器；D2 用于验证单基因扰动预测及其谱系程序方向。

由于 Perturb-seq 终点测量会破坏细胞，D2 没有同一细胞的扰动前状态。因此本轮可以评价群体分布和背景条件泛化，不能把个体起始状态闸门写成通过；相应状态应为 `MODEL_STATE_VALID=NOT_EVALUABLE`，不得伪造为 `true`。

## 2. 已完成的前置核对

- D2 三个条件完整 CSR 审计通过，固定摘要哈希：`c7c2f4f7116a625c690e5370b8fcc9ceabac0a3b3ad2eb385b9c4a9a66eee7c5`。
- 三个条件均为18,130个特征，其中18,129个Ensembl基因和一个 `PuroR`；三者基因轴一致。
- 36个预审计 Naive/Th1/Th2/Th17 候选基因全部存在于三个D2文件的测量轴。
- 17,033个细胞的HVG预审计结果：Naive 8/8、Th1 5/6、Th2 5/9、Th17 13/13；Th2身份基因 `GATA3` 和 `CCR4` 有进入正式top 2,000之外的风险。
- 当前 `config/programs.json` 仍为空，尚未形成可用于模型验收的冻结评分程序。

证据文件：

- `research/state_d2/th_lineage_marker_evidence.md`
- `research/state_d2/d2_hvg_marker_precheck.json`

## 3. 阶段A：冻结并验证谱系评分器

### A1. 冻结程序层级

先新增 `config/th_lineage_programs.v1.json`，基因按功能分层，禁止混成一个分数：

- Naive身份：`TCF7 LEF1 CCR7 SELL MAL`
- Th1身份：`TBX21 IL12RB2 CXCR3`
- Th2身份：`GATA3 CCR4 PTGDR2`
- Th17身份：`RORC CCR6 IL23R`
- Th1效应确认：`IFNG`
- Th2效应确认：`IL4 IL5 IL13`
- Th17效应确认：`IL17A IL17F CCL20 IL26`
- 泛激活、应激、凋亡/存活、细胞周期、干扰素反应分别建立独立程序。

初始版本中，同一层级内使用等权重，不根据D2扰动响应调整权重。每个程序同时保存基因符号、Ensembl ID、文献来源、适用时间和局限性。

### A2. 建立两个独立的外部验证集

1. `GSE135390`：外周血直接分选的Naive、Th1、Th2、Th17等人CD4亚群，三位供者；用于验证成熟亚群可分性。
2. Cano-Gamez等2020年人初始CD4极化数据：比较Rest/Th0与Th1、Th2、Th17，并区分早期激活和约第5天成熟程序；公开计数表优先，受限原始数据不是本轮必要条件。

两套参考均不得参与D2的HVG计算、模型训练或迁移参数选择。若第二套公开计数表无法稳定取得，必须记录失败原因，并以另一套独立、同来源人初始CD4极化数据替换；不得只保留一个验证集后仍声称“独立验证”。

### A3. 评分和通过标准

跨平台评分采用样本内基因百分位秩，身份分数和效应分数分别报告；不直接比较不同数据集的原始表达量。

成熟期硬标准：

1. 对每个Th谱系，其身份分数在对应亚群中高于Naive和两个非目标谱系，至少在三位供者中的两位成立；
2. 三个Th身份分数的一对多宏平均受试者工作特征曲线下面积不低于0.80；
3. Naive身份分数在Naive中高于三个成熟Th亚群，至少三位供者中的两位成立；
4. 泛激活或细胞周期分数不能单独复现谱系分类结果。

早期时间点只作诊断，不因 `IL4/IL5/IL13/IL17A/IL17F` 未充分出现而判失败。

输出：

- `research/state_d2/program_validation/reference_manifest.json`
- `research/state_d2/program_validation/program_scores.tsv`
- `research/state_d2/program_validation/program_validation_report.md`

失败处理：若身份分数未通过，只能依据文献和第一套开发参考修订一次，然后用第二套参考锁定验证；第二套仍失败则停止D2程序方向闸门，但可以继续做一般表达预测基准。

## 4. 阶段B：D2数据合同和扰动词表

### B1. 建立新的运行目录

远端使用：

```text
/root/autodl-tmp/CRISPR_perturb_runtime/pipeline/development_D2_v2/
```

不得覆盖旧的 `development_D2_v1`、原始 `.h5ad` 或既有审计结果。运行前检查数据盘可用空间；低于220 GiB立即停止新增大型产物。

### B2. 激活D2角色并完成数据审计

使用已经通过的2026-09-05 CSR摘要运行 `activate-roles`。随后只读记录：

- 三个文件的细胞数、基因数、文件指纹；
- `.X` 是否为整数原始计数，以及 `.raw` 和各layer的状态；
- `gene_name`、Ensembl ID及唯一性；
- guide、扰动、非靶向对照、低质量、多guide、无guide标签；
- lane、run和技术批次字段；
- 三个生物学条件的准确命名。

输出：`research/state_d2/d2_data_audit.json`。

通过标准：三条件基因轴完全一致；CSR摘要哈希吻合；原始矩阵非负、有限且数据语义明确；`PuroR` 不进入蛋白编码基因空间。

### B3. 冻结扰动词表和敲低质量

先按固定guide库校订标签，再执行：

- `low_quality=false`；
- 仅单sgRNA；
- non-targeting对照单独保留；
- 多guide、无guide和标签冲突分开报告；
- 多个guide指向同一基因时，共用一个扰动身份，但保留guide层级用于自助法和一致性评价。

按照STATE论文遗传扰动规则，在CP10K+log1p尺度计算主要敲低过滤：扰动均值残余表达不高于0.30、逐细胞残余表达不高于0.50、过滤后不少于30个细胞；原始计数尺度只生成敏感性报告，不改变主要纳入结果。

输出：

- `research/state_d2/d2_perturbation_vocab.json`
- `research/state_d2/d2_guide_qc.json`
- `research/state_d2/d2_knockdown_filter.json`

## 5. 阶段C：先划分，再计算正式HVG

### C1. 冻结主要划分

种子固定为 `20260901`。主要任务是基因×生物背景留出：

1. 对每个入选扰动基因，使用稳定哈希分配一个目标条件；
2. 该基因在另外两个条件的响应可进入训练，在目标条件的响应完全进入测试；
3. 从训练组合中再稳定划出验证组合；
4. non-targeting对照可在三个条件中用于建立背景分布，但控制细胞ID不能跨集合重复；
5. 同一 `(perturbation, condition)` 不得同时进入训练、验证和测试。

另建两个诊断划分：随机细胞划分和基因×NTC局部状态留出。局部状态必须由NTC建立，只能解释为匹配的群体状态区域，不能解释为每个扰动细胞真实的扰动前状态。

输出：`research/state_d2/splits/*.json`，每份包含细胞选择规则、组合列表、种子和SHA-256。

### C2. 正式流式HVG计算

只在主要划分的训练组合和允许使用的NTC控制中拟合基因统计量：

1. 使用固定人蛋白编码基因清单与D2实测Ensembl基因的交集；
2. 排除 `PuroR`；
3. 每个细胞在保留基因交集上重新计算总计数；
4. `normalize_total(target_sum=10000)`；
5. `log1p`；
6. 使用Scanpy 1.11.5、`flavor=seurat`、`n_bins=20`、`batch_key=None`计算原始top 2,000。

实现必须流式读取CSR，不得把约880万×18130矩阵整体载入或整体稠密化。使用一个可人工构造的小矩阵与Scanpy直接计算逐项对照，数值容差写入测试。

### C3. 身份锚点替换规则

先报告纯top 2,000覆盖，再仅对14个预注册身份锚点判断是否有资格强制纳入。资格必须同时满足：

- 三个条件均测得，Ensembl与gene symbol一对一；
- 属于冻结的人蛋白编码交集；
- 在训练部分至少两个条件的检测率不低于0.5%；
- 训练部分总检测细胞不少于500；
- 标准化离散度名次不低于5,000名；
- 不是近乎全零或映射含糊基因。

每加入一个合格缺失锚点，删除一个名次最靠后的非强制HVG。最多允许10个强制基因；超过即停止冻结并重新审查目标空间，禁止静默塞入大量标记。效应细胞因子和STAT总表达不因文献重要性而自动强制纳入。

输出：`research/state_d2/d2_gene_panel_2000.json`，必须包含：

- 2,000个有序gene symbol和Ensembl ID；
- 原始HVG名次与标准化离散度；
- 每个程序基因的检测率和均值；
- `forced_include`、`replaced_genes`和理由；
- 预处理参数、输入划分哈希和面板SHA-256。

Scratch、Transfer、简单基线和同一次评价必须使用完全相同的面板顺序。运行时硬断言模型基因顺序与面板哈希一致。

## 6. 阶段D：先做64–100个扰动的小型试点

试点选择只使用guide支持度、敲低质量、对照表达和文献预注册类别，不使用最终测试性能。优先纳入通过敲低过滤且细胞量足够的 `TBX21`、`GATA3`、`RORC`、`STAT4`、`STAT6`、`STAT3`、`BATF`、`IRF4`；未通过敲低过滤者标记为不可评价，不为凑齐机制基因而降低标准。

当前obs计数表明TBX21、GATA3、RORC、STAT4、STAT6在三个条件均有约百至数百个基础质控合格细胞，具备进入后续敲低质量审计的可行性；IFNG仅0–6个，不作为稳定试点靶点。

首先运行 `STATE-D2-Scratch-Pilot`。固定集合大小 `S=32`，结构沿用官方Replogle的隐藏维度128、4层编码器、4层解码器和8个注意力头，不使用batch encoder；Rest、Stim8hr、Stim48hr作为生物背景，lane/run只保留为技术元数据。

试点硬标准：

- 数据加载器不稠密化全量矩阵；
- 训练损失有限并下降；
- 输出严格为 `S×2000`；
- 输出顺序和面板哈希完全一致；
- 所有扰动都存在于冻结词表，无静默回退；
- 无NaN/Inf；
- 显存稳定，不超过32 GiB；
- 可以保存、重新加载并得到一致预测。

输出：

- `research/state_d2/pilot/pilot_manifest.json`
- `research/state_d2/pilot/pilot_metrics.json`
- 远端检查点目录 `models/state_d2_pilot/`

任一硬断言失败时停止全量训练。

## 7. 阶段E：Scratch和Transfer公平训练

### E1. 共同训练合同

- 随机种子：`20260901、20260902、20260903`；
- 每个模型每个种子最多40,000步；
- 使用验证MMD早停并保存最佳检查点；
- 不以训练损失选择模型；
- 使用相同细胞池、集合采样、划分、2,000基因、扰动词表和总训练预算；
- 试点确定批大小后冻结；OOM只能降低共同批大小并重跑双方，不得只优待Transfer。

### E2. Scratch

所有参数随机初始化，结构与试点通过版本一致。输出到远端 `models/state_d2_scratch/`。

### E3. Transfer

实例化完全相同的D2模型，再从官方 `ST-HVG-Replogle` 按名称和张量语义迁移：

- 形状和语义都一致的Transformer注意力、前馈层、LayerNorm和hidden-to-hidden参数完整复制；
- 输入和输出层按gene symbol对齐，只复制最终D2面板的重叠基因位置；
- 扰动层按名称且语义一致时复制，绝不按整数索引复制；
- D2独有基因、D2新扰动、背景相关参数随机初始化；
- 无法确认生物学轴的张量不复制，并记录原因。

先生成 `research/state_d2/transfer_report.json`，通过人工和自动断言后才训练。

Transfer分两段，但总步数不得超过Scratch：

1. 冻结Transformer主干，训练新输入/输出和扰动参数；
2. 解冻主干，新参数学习率高于主干学习率，继续使用验证早停。

输出到远端 `models/state_d2_transfer/`。

## 8. 阶段F：统一评价和生物学方向验证

### F1. 简单基线

同一面板和划分运行：

- no-change；
- condition mean；
- perturbation mean；
- 低秩线性残差模型；
- `pert2state`，仅在接口、基因顺序和划分完全一致时纳入。

### F2. 主要模型指标

- 绝对伪总体表达变化的Pearson相关；
- 基于曼哈顿距离的扰动区分指标；
- 伪总体MAE/MSE；
- MMD及分布层评价；
- Wilcoxon加BH校正后的DE方向、效应量秩相关、top-k重叠；
- Rest、Stim8hr、Stim48hr分别报告；
- 三个随机种子和基因×背景留出分别报告；
- NTC空效应误差和不确定性校准。

主要全局闸门：STATE相对最佳简单基线的伪总体变化Pearson绝对提升至少0.05，且分层自助法差值95%置信区间下界大于0；扰动区分指标下降不得超过0.02；MAE和任何单一条件不得恶化超过10%；NTC误差不得超过NTC空模型第95百分位。

### F3. Naive/Th1/Th2/Th17方向验证

评分器先在外部极化数据通过后，才用于D2预测。对每个通过敲低过滤的机制基因，同时比较真实D2响应和STATE预测：

- `TBX21`敲低：Th1身份/效应程序预期减弱；
- `GATA3`敲低：Th2身份/效应程序预期减弱；
- `RORC`敲低：Th17身份/效应程序预期减弱；
- STAT家族只按相应通路方向作辅助解释，不以总转录本替代磷酸化活性。

只有在对应背景中程序表达可检测、on-target敲低通过且真实D2方向显著时，才对预测方向计分；否则记为 `NOT_EVALUABLE`，不得记成模型失败或成功。

这一步验证的是“单基因扰动能否正确推动或削弱已冻结的谱系程序”，不是证明D2细胞已经完成Naive→Th1/Th2/Th17极化。

## 9. 阶段G：模型选择和停止规则

生成 `research/state_d2/state_model_comparison.md`，只允许以下结论：

1. `STATE-D2-Transfer`：主要留出指标稳定优于Scratch，至少三个种子中的两个方向一致，并且各条件没有明显负迁移；
2. `STATE-D2-Scratch`：Scratch更好，或两者差异不稳定但Scratch更稳健；
3. `STATE未通过`：两者都没有超过简单基线，设置 `MODEL_GLOBAL_VALID=false` 并停止；
4. `程序方向不可评价`：外部评分器或D2机制读出不满足条件，保留一般表达预测结论，但不声明谱系方向通过。

即使群体模型通过，`MODEL_STATE_VALID` 在当前终点Perturb-seq设计下仍保持 `NOT_EVALUABLE`。因此本轮结束后不自动进入组合性和序贯规划。

## 10. 测试、复现和Git提交

每个阶段至少包含：小型合成数据单元测试、命令级冒烟测试、JSON结构验证、哈希复算和远端 `pip check`。大型任务只保存精简清单、指标和哈希到Git；原始 `.h5ad`、权重和大矩阵只保留在服务器。

只使用一个大目标分支：

```text
d2-state-single-step
```

建议提交顺序：

1. `初始化项目并建立安全提交规则`
2. `冻结辅助T细胞评分程序并完成外部验证`
3. `实现D2审计、划分和2000基因面板`
4. `完成D2 STATE试点和统一基线`
5. `完成STATE从头训练与迁移训练`
6. `完成统一评估并冻结模型结论`

每次提交前执行：

```powershell
git status --short
python -m pytest
git diff --check
git diff --cached --name-only
```

检查通过后使用中文提交信息并推送当前分支。禁止强制推送。提交中不得包含 `connect_server.py`、`remote_config.json`、访问令牌、原始数据或模型权重。

## 11. 最终交付清单

1. `d2_data_audit.json`
2. `th_lineage_programs.v1.json`及两套外部验证报告
3. `d2_perturbation_vocab.json`
4. `d2_guide_qc.json`和`d2_knockdown_filter.json`
5. 冻结划分文件及哈希
6. `d2_gene_panel_2000.json`
7. `transfer_report.json`
8. Scratch与Transfer三个种子的远端检查点清单和哈希
9. 全部简单基线与统一指标
10. `state_model_comparison.md`
11. `research/state_d2/README.md`，记录环境、命令、Git提交、随机种子、远端路径和复现步骤
12. 明确结论：Scratch、Transfer、STATE未通过，或程序方向不可评价

执行顺序不可跳过：评分器冻结与外部验收 → D2数据合同 → 扰动词表和敲低过滤 → 划分 → 正式2,000基因面板 → Pilot → Scratch/Transfer → 统一评价 → 冻结结论。
