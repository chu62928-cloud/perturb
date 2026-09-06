# 人初始 CD4 T 细胞向 Th1、Th2、Th17 分化的转录程序标记证据

更新日期：2026-09-06

## 结论先行

1. 文献可以预先冻结一份“必须检查是否进入 D2 top 2,000 HVG”的基因清单，但**不能从文献推断这些基因一定会进入 D2 的 top 2,000 HVG**。最终是否入选必须在冻结的数据划分上实际计算，并同时报告是否测得、训练集检测率、均值、离散度和 HVG 名次。
2. 初始细胞和三条辅助 T 细胞谱系都不能由一个基因可靠判定。建议将“谱系决定转录因子/受体”“刺激后效应因子”和“竞争谱系/混杂程序”分层计分。
3. 用于最初验证的合适任务是同来源人初始 CD4 T 细胞在匹配的 Th0、Th1、Th2、Th17 极化条件下比较；16 小时以内主要反映共同激活，约 5 天后才更适合评价谱系程序。Cano-Gamez 等直接比较了人初始 CD4 T 细胞在 16 小时和 5 天的 Th0、Th1、Th2、Th17 等条件，发现早期变化主要由 TCR/CD28 激活主导，而多数细胞因子诱导的谱系变化到第 5 天才清楚出现。[Cano-Gamez 等，Nature Communications，2020](https://www.nature.com/articles/s41467-020-15543-y)
4. D2 的 Rest、Stim8hr、Stim48hr 是刺激背景，不是已经确认的 naive、Th1、Th2、Th17 极化标签。因此，D2 内部只能验证“向某谱系程序移动”的方向，不能仅凭分数把细胞重新命名为相应谱系。

## 建议冻结的 HVG 覆盖审计清单

下表中的“身份锚点”应优先检查；“效应确认”常受培养时长和再刺激影响，不应因其低表达就单独判定极化失败，也不宜不加条件地全部强制塞进 2,000 维面板。

| 程序 | 身份锚点：优先审计 | 效应确认：条件依赖 | 辅助标记及主要局限 |
|---|---|---|---|
| 初始/naive | `TCF7, LEF1, CCR7, SELL, MAL` | 不设单一效应因子 | `LRRN3, IL7R, KLF2, BACH2` 可辅助；`IL7R` 也见于记忆细胞，`CCR7/SELL` 会受激活影响 |
| Th1 | `TBX21, IL12RB2, CXCR3` | `IFNG` | `IL18R1, STAT4, HLX` 可辅助；`STAT4` 的表达量不能替代其磷酸化活性，`IFNG` 可被短时共同激活诱导 |
| Th2 | `GATA3`，并联合 `CCR4` 或 `PTGDR2` | `IL4, IL5, IL13` | `IL1RL1` 可作成熟/组织型 Th2 辅助标记；`CCR4` 可在激活 Th1 和 Th17 中出现，`PTGDR2` 会在 TCR 激活后下降 |
| Th17 | `RORC, CCR6, IL23R` | `IL17A, IL17F, CCL20, IL26` | `KLRB1, IL22` 可辅助；`KLRB1` 不只属于 Th17，`IL22` 也可代表 Th22，效应细胞因子常稀疏且依赖再刺激 |

### 最小“身份锚点”集合

如果项目必须在保持恰好 2,000 个基因的前提下考虑少量文献驱动替换，建议首先只对下面这组执行“是否有资格强制纳入”的审计，而不是直接强制全部纳入：

```text
TCF7 LEF1 CCR7 SELL MAL
TBX21 IL12RB2 CXCR3
GATA3 CCR4 PTGDR2
RORC CCR6 IL23R
```

资格条件应至少包括：D2 确实测得该基因、属于冻结的人蛋白编码基因交集、在训练部分达到预先规定的检测率或表达阈值，且不存在名称/Ensembl 映射歧义。没有测得或在训练集中近乎全零的基因不能靠“强制纳入”恢复信息。

### 效应确认集合

下面这组适合在成熟期或标准化再刺激后作外部生物学确认；应报告其 HVG 覆盖，但不建议仅因文献重要性就无条件替换 HVG：

```text
IFNG IL4 IL5 IL13 IL17A IL17F CCL20 IL26
```

## 各谱系证据与判读边界

### 初始/naive 程序

- 人初始 CD4 T 细胞在静息状态可由 `SELL`、`CCR7` 和 `LRRN3` 等高表达标记定位；从初始到记忆/效应的连续变化伴随这些标记下降和细胞因子、趋化因子上升。[Cano-Gamez 等，2020](https://www.nature.com/articles/s41467-020-15543-y)
- `MAL` 在人初始 CD4 T 细胞中高表达，并随激活及向中央/效应记忆分化逐步降低，可作为独立的初始状态辅助锚点。[MAL 人 CD4 研究](https://pubmed.ncbi.nlm.nih.gov/33345332/)
- `TCF7`、`LEF1`、`CCR7`、`SELL` 是常用的低效应化/初始程序组合，但 `CCR7/SELL` 在激活后会变化，不能拿单个时间点的下降量直接等同于获得 Th1、Th2 或 Th17 身份。[Cano-Gamez 等，2020](https://www.nature.com/articles/s41467-020-15543-y)

因此，naive→ThX 的正确预期应是“naive 多基因程序下降，同时目标谱系多基因程序上升”，而不是只看 `SELL` 或 `CCR7` 下降。

### Th1 程序

- `TBX21` 编码 T-bet，是 Th1 谱系决定因子；原始研究显示 T-bet 能启动 `IFNG`，并压低 `IL4/IL5` 等竞争的 Th2 效应程序。[Szabo 等，Cell，2000](https://pubmed.ncbi.nlm.nih.gov/10761931/)
- 在人初始 CD4 T 细胞中，`IL12RB2` 是发育中的 Th1 细胞响应 IL-12 的关键受体亚基；TCR 诱导的染色质重塑和 IL-12/STAT4 信号共同促进其表达。[人 Th1 的 IL12RB2 调控研究](https://pubmed.ncbi.nlm.nih.gov/17304212/)
- 获得性 STAT4 缺陷的人 CD4 T 细胞在 Th1 培养中 `IFNG`、`IL12RB2` 和 `TNF` 明显降低，支持 `IL12RB2` 与 `IFNG` 作为人 Th1 程序读出；但 `TNF` 过于泛化，不宜列入核心身份分数。[人 STAT4 缺陷研究](https://pubmed.ncbi.nlm.nih.gov/19359411/)
- `CXCR3` 常富集于人 Th1，但趋化受体表达具有可塑性；它应与 `TBX21/IL12RB2` 联合，而不能单独判型。[Sallusto 等，JEM，1998](https://pubmed.ncbi.nlm.nih.gov/9500790/)
- 人初始 CD4 T 细胞的早期 Th1 时间序列显示，`CD69` 约 0.5 小时即出现，`IL2`、`IFNG`、`TBX21` 约 2 小时开始增加；`FOS/JUN` 等共同激活因子随后强烈上升。这说明 8 小时左右的 `IFNG` 或 `TBX21` 上升仍需 Th0/共同激活对照，不能单基因定性。[早期人 Th1 RNA-seq/ATAC-seq](https://pmc.ncbi.nlm.nih.gov/articles/PMC8848251/)

### Th2 程序

- 人 GATA3 单倍剂量不足者的 Th2 频率和 Th2 功能降低；在正常人 CD4 T 细胞中敲低 GATA3 也抑制 Th2 分化，支持 `GATA3` 作为最重要的人 Th2 身份锚点。[GATA3 人 Th2 研究](https://pubmed.ncbi.nlm.nih.gov/14757746/)
- `IL4`、`IL5`、`IL13` 是成熟 Th2 的效应输出，但这些转录本在未再刺激的单细胞数据中可能稀疏；文献中的功能判定经常在极化后再用 PMA/ionomycin 或 TCR 刺激检测。因此它们适合组成“效应确认分数”，不适合作为唯一身份判据。[GATA3 人 Th2 研究](https://pubmed.ncbi.nlm.nih.gov/14757746/)
- `CCR4` 富集于人 Th2，但激活后的 Th1 也可上调 `CCR4`，而部分 Th17 同样为 `CCR4+`；所以 `CCR4` 只能作为组合标记。[CCR4/CCR8 激活研究](https://pubmed.ncbi.nlm.nih.gov/9820476/)
- `PTGDR2`（CRTH2）可支持成熟人 Th2 身份，但 TCR/CD28 激活 24 小时会降低其转录和表面表达，因而 Stim8hr/Stim48hr 阶段的低值不能直接作为反证。[人 Th2 的 CRTH2 调控研究](https://pubmed.ncbi.nlm.nih.gov/29969451/)

### Th17 程序

- 人 Th17 记忆细胞中，`CCR6+CCR4+` 细胞富集 IL-17 产生和 `RORC`；`CCR6+CXCR3+` 群体则可同时包含 IFN-γ 与 IL-17 程序。[Acosta-Rodriguez 等，Nature Immunology，2007](https://pubmed.ncbi.nlm.nih.gov/17486092/)
- 人初始 CD4 T 细胞的极化实验表明，RORγt（`RORC`）居于人 Th17 分化中心，并伴随 `CCR6`、`IL23R`、`IL17F`、`IL26` 等表达；`IL17A` 的有效诱导取决于细胞因子组合和培养条件。[Manel 等，Nature Immunology，2008](https://pubmed.ncbi.nlm.nih.gov/18454151/)
- 人 Th17 和 Th17/Th1 克隆均可表达 `IL23R`、`CCR6`、`RORC`，并且部分同时表达 T-bet 和 IFN-γ；IL-12 又能推动其向 Th1 样方向变化。因此 `TBX21` 或 `IFNG` 不能作为排除 Th17 的硬性负标记。[Annunziato 等，JEM，2007](https://pubmed.ncbi.nlm.nih.gov/17635957/)
- 独立的人转录组研究发现，`IL17A` 和 `CCL20` 对 Th17 富集群较特异，但在非 Th17 群接近零；相反，`IL23R` 也可见于 Th1，说明应联合多基因而非依赖单一受体。[Zhang 等，PLoS One，2012](https://pubmed.ncbi.nlm.nih.gov/22715389/)
- `FOXP3` 也不能作为排除 Th17 的绝对负标记：人组织和外周血中存在能产生 IL-17 的 `FOXP3+CCR6+` 调节性 T 细胞。[人 IL-17+ FOXP3+ Treg 研究](https://pubmed.ncbi.nlm.nih.gov/19273860/)

## “负向标记”应如何使用

建议不要维护一份声称在所有时间和背景都成立的硬负向基因表，而是计算彼此独立的竞争程序分数：

- Th1 结果同时报告 Th2 与 Th17 程序；
- Th2 结果同时报告 Th1 与 Th17 程序；
- Th17 结果同时报告经典 Th1、Th2 和 Treg 程序；
- naive 结果同时报告三条效应谱系以及共同激活程序。

可作为**软性竞争程序**审计的基因如下：

```text
Th1竞争程序: TBX21 IL12RB2 CXCR3 IFNG
Th2竞争程序: GATA3 CCR4 PTGDR2 IL4 IL5 IL13
Th17竞争程序: RORC CCR6 IL23R IL17A IL17F CCL20 IL26
Treg竞争程序: FOXP3 IL2RA CTLA4 IKZF2
```

它们用于比较分数方向，不用于“一旦出现就否决另一谱系”。这是因为人 Th17/Th1 混合状态、Th17/Treg 可塑性以及趋化受体共享均有直接证据。

## 必须分开的混杂程序

### 泛激活

```text
FOS JUN JUNB FOSL1 FOSL2 EGR1 EGR2 EGR3 NR4A1 CD69 IL2 IL2RA TNFRSF4 ICOS
```

这些基因主要报告 TCR/CD28 反应、AP-1/即时早期反应或共同激活，不应被吸收到 Th1/Th2/Th17 身份分数中。人初始 CD4 T 细胞早期 Th1 培养中，`CD69`、`FOS/JUN` 和 `IL2` 的变化先于或伴随谱系变化；更大规模的人极化研究也显示早期表达首先由共同激活主导。[早期人 Th1 时间序列](https://pmc.ncbi.nlm.nih.gov/articles/PMC8848251/)；[Cano-Gamez 等，2020](https://www.nature.com/articles/s41467-020-15543-y)

### 应激

```text
HSPA1A HSPA1B HSPA2 HSPA6 DNAJB1 GADD45B IER2
```

人 CD4 T 细胞抗 CD3/CD28 刺激的单细胞研究分辨出独立的高热休克蛋白状态，特征包括 `HSPA1A/HSPA1B/HSPA2/HSPA6/DNAJB1/GADD45B/IER2`；该状态同时富集应激、未折叠蛋白和凋亡相关过程，不能当作任何一条辅助 T 谱系。[Zhang 等，iScience，2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10460988/)

### 凋亡/存活选择

```text
BBC3 CASP3 BCL2A1 TNFRSF10A TNFRSF10B FAS FASLG PMAIP1 BCL2L11
```

人原代 CD4/CD8 T 细胞激活的时间转录组显示，BCL2、CASPASE 和 TNF 受体家族会随激活而动态改变；其中 `BCL2A1`、`BBC3`、`CASP3` 还得到蛋白层验证。因此这组基因应作为“凋亡/存活选择”单独评分，不应被解释成谱系分化。[Wang 等，BMC Medical Genomics，2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2600644/)

### 另外建议报告但不并入谱系分数

- 细胞周期：`MKI67, TOP2A, PCNA, TYMS, STMN1`；
- I 型干扰素反应：`ISG15, IFIT1, IFIT3, MX1, MX2, OAS1, OAS2, OAS3`；
- 细胞毒/高效应化：`NKG7, GNLY, PRF1, GZMB, CCL5`。

这些程序可能因刺激背景、培养时间或亚群组成改变而上升。尤其 `NKG7/GNLY/PRF1/GZMB` 更接近细胞毒/高效应化，不宜充当普通 CD4 Th1 核心分数。Cano-Gamez 等在人 CD4 T 细胞中观察到激活后细胞周期、I 型干扰素和效应化差异，并将初始到高效应记忆描述为连续梯度。[Cano-Gamez 等，2020](https://www.nature.com/articles/s41467-020-15543-y)

## 对 D2 top 2,000 HVG 审计的可执行要求

在看 D2 响应或模型结果之前冻结上述清单，然后对每个基因输出以下字段：

```text
gene_symbol
ensembl_gene_id
program
marker_tier               # identity_anchor / effector_confirmation / confounder
measured_in_all_conditions
training_detection_rate
training_mean_log1p_cp10k
training_dispersion
hvg_rank
in_top_2000
force_eligible
decision_reason
```

推荐决策顺序：

1. 未测得、非目标蛋白编码基因或映射含糊：记录缺失，不得补零或强制加入。
2. 已测得但训练集近乎全零：保留为外部蛋白/细胞因子验证候选，不强制加入表达模型。
3. 身份锚点在训练集可靠表达、只因 HVG 名次略低于 2,000 而缺失：可在预先冻结的规则下少量替换末位 HVG，并记录原始名次和被替换基因。
4. 效应确认基因缺失：首先解释其阶段和再刺激依赖性；不因单个细胞因子缺失就改写整个面板。
5. Scratch 与 Transfer 必须使用同一最终顺序；文献标记仅决定覆盖审计与预先规定的替换规则，不参与模型优劣选择。

## 建议的首轮简单验证

1. 使用独立的人初始 CD4 极化参考，建立同供者的 Rest/Th0/Th1/Th2/Th17 比较；Cano-Gamez 2020 的 16 小时与 5 天设计可同时检验“早期激活”和“成熟谱系”能否被分开。
2. 主要判据放在第 5 天或相近成熟时间：目标身份锚点与效应确认分数共同上升，naive 程序下降，并且目标谱系相对匹配 Th0 有额外增益。
3. 16 小时读出只评价早期方向，不要求 `IL4/IL5/IL13/IL17A/IL17F` 全部出现；泛激活分数必须单独报告。
4. 每条谱系至少需要多基因方向、整体转录组相似度和独立的蛋白/细胞因子读出中的两类证据；单一 `IFNG`、`GATA3` 或 `IL17A` 不足以宣布分化成功。
5. 把外部参考用于验收评分器，不用同一参考同时选择基因、调阈值和宣称独立验证。D2 只用于检验扰动是否沿这些冻结程序移动。

## 供实现直接读取的基因清单

```yaml
naive_identity:
  - TCF7
  - LEF1
  - CCR7
  - SELL
  - MAL
naive_supporting:
  - LRRN3
  - IL7R
  - KLF2
  - BACH2
th1_identity:
  - TBX21
  - IL12RB2
  - CXCR3
th1_effector:
  - IFNG
th1_supporting:
  - IL18R1
  - STAT4
  - HLX
th2_identity:
  - GATA3
  - CCR4
  - PTGDR2
th2_effector:
  - IL4
  - IL5
  - IL13
th2_supporting:
  - IL1RL1
th17_identity:
  - RORC
  - CCR6
  - IL23R
th17_effector:
  - IL17A
  - IL17F
  - CCL20
  - IL26
th17_supporting:
  - KLRB1
  - IL22
general_activation:
  - FOS
  - JUN
  - JUNB
  - FOSL1
  - FOSL2
  - EGR1
  - EGR2
  - EGR3
  - NR4A1
  - CD69
  - IL2
  - IL2RA
  - TNFRSF4
  - ICOS
stress:
  - HSPA1A
  - HSPA1B
  - HSPA2
  - HSPA6
  - DNAJB1
  - GADD45B
  - IER2
apoptosis_survival_selection:
  - BBC3
  - CASP3
  - BCL2A1
  - TNFRSF10A
  - TNFRSF10B
  - FAS
  - FASLG
  - PMAIP1
  - BCL2L11
```

## 证据边界

- 上述清单是面向人初始 CD4 T 细胞和人 Th 亚群文献整理的**预注册审计候选**，不是未经 D2 检验就成立的最终面板。
- 不同研究采用外周血或脐带血、不同细胞因子组合、培养时间和再刺激方案；效应细胞因子的可检测性不能跨实验条件直接等同。
- 谱系决定因子具有因果证据，但其转录本丰度仍不等于蛋白活性或稳定命运；STAT 家族尤其需要区分总表达与磷酸化信号。
- 人 Th17 具有明显的 Th17/Th1 和 Th17/Treg 可塑性，因此竞争程序适合做连续分数，不适合做绝对互斥标签。
