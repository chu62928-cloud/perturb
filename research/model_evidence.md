# STATE、Stack 与 CellFlow：可行性方案证据核查

核查日期：2026-08-31。仅采用作者论文、官方仓库、官方文档及预印本平台元数据；未训练、未安装模型、未下载权重或数据、未写入服务器。本文使用 research 技能的一手来源核查规范，并按用户要求直接完成、不委派代理。硬件与数据条件来自用户本轮提供的信息，不冒充本次实测。

## 结论与适用范围

- **STATE 小型状态转移模型（ST）可列为单终点扰动预测的候选基线，但目前不能声称已验证 Th2 跨环境预测。** 主要论文实验含目标环境的少量扰动监督；默认独热编码也不自动支持完全未见的靶基因。[论文第 2.2、4.2、4.3 节](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full)
- **Stack 可用于基于表达谱提示的跨环境假设生成，不能在没有 Th2 扰动真值时称作 Th2 校准器。** 不输入类别标签、不微调，与不需要提示中的响应信息是不同概念。[官方预测示例](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/notebooks/tutorial-predict.ipynb)
- **CellFlow 对组合干预、时间条件和多阶段培养方案有直接证据，值得作为方法候选；这些证据仍不等于序贯 CRISPRi 已解决。** 论文使用处理起止时间编码预测培养方案结果，没有据此证明任意基因 A→B 与 B→A 的干预顺序效应。[论文图 6 及类器官方法](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full)

本文的 CellFlow 特指 **theislab/CellFlow**，不混入其他同名影像生成项目。[原始代码与论文入口](https://github.com/theislab/CellFlow)

## 1. 论文日期、状态与代码定位

| 模型 | 已核实的论文与发布日期 | 状态及代码 |
|---|---|---|
| STATE（Arc） | *Predicting cellular responses to perturbation across diverse contexts with State*；v1 为 **2025-06-27**，v2 为 **2025-07-10**。DOI 中的 06-26 不是首次公开日期。[平台元数据](https://api.biorxiv.org/details/biorxiv/10.1101/2025.06.26.661135) | 本次确认到 bioRxiv 预印本；平台 `published=NA`，未核实正式期刊版本。[论文](https://doi.org/10.1101/2025.06.26.661135)、[原始代码](https://github.com/ArcInstitute/state) |
| Stack（Arc） | *Stack: In-Context Learning of Single-Cell Biology*；v1 为 **2026-01-09**，v2 为 **2026-06-08**。[平台元数据](https://api.biorxiv.org/details/biorxiv/10.64898/2026.01.09.698608) | 本次确认到 bioRxiv 预印本，`published=NA`。PMC 收录的可读全文为 v1，不能当作已发表期刊论文。[v2](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v2.full)、[v1 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803207/)、[代码](https://github.com/ArcInstitute/stack) |
| CellFlow（Theis 团队） | *CellFlow enables generative single-cell phenotype modeling with flow matching*；v1 为 **2025-04-17**，不是 DOI 中的 04-11。[平台元数据](https://api.biorxiv.org/details/biorxiv/10.1101/2025.04.11.648220) | 本次平台记录只返回 v1，`published=NA`；未核实正式期刊版本。[论文](https://doi.org/10.1101/2025.04.11.648220)、[代码](https://github.com/theislab/CellFlow)、[文档](https://cellflow.readthedocs.io/) |

“未核实正式期刊版本”不等于断言从未接收；不采用作者投稿计划或第三方文章替代出版记录。STATE 的 Arc 发布公告日期为 2025-06-23，应与论文公开日期区分。[Arc 公告](https://arcinstitute.org/news/virtual-cell-model-state)

源码快照：STATE `9bbfe78a434a55205e4de834e1ea99f85f7a3add`；Stack `cacc2e4b09435c3e536d46237d10b50f222dd144`；CellFlow `446ed6073c60ac2e8db13c4ea096a43cdec288b2`。下列关键代码链接固定到本次检查的提交，避免把后续变化误当作本文证据。

## 2. STATE：有监督的群体响应预测，不是通用未见基因模拟器

**任务边界。** 原文评估了药物、细胞因子和遗传扰动，包括 Replogle–Nadig 遗传扰动任务。输入为对照细胞集合及扰动表示，预测处理后表达分布；没有同一细胞的连续实测轨迹。论文将组合扰动扩展列为后续方向，不应把官网概括性宣传当作序贯基因干预验证。[论文第 2、3、4.3 节](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full)

**“保留 30%”核查：成立，但须限定实验。** v1、v2 的欠代表环境泛化实验均明确让训练过程看到测试环境中 **30% 的扰动条件**；Replogle–Nadig 的留一细胞系实验使用另外三个细胞系，加上目标细胞系的这 30%，再预测剩余条件。这不是完全没有目标扰动数据的留出细胞系，也不是随机留下 30% 的细胞。论文同时存在严格零样本环境任务：预测其他环境已经观察过的同一扰动在新环境中的效应；v2 还明确它没有验证整个查询数据集的全部环境都未见过的跨数据集迁移。不要混用这些成绩。[v1 第 2.2、4.2 节](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1.full)、[v2 第 2.3、3、4.2 节](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full)

**未见环境不等于未见基因。** 当前默认 `pert_rep: onehot`；论文的零样本任务是已见扰动迁移至新环境。源码允许通过 `perturbation_features_file` 引入外部特征，但“存在接口”不能证明对完全未训练靶基因已具备有效泛化。[默认数据配置](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/src/state/configs/data/perturbation.yaml)、[论文第 4.2–4.3 节](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full)

**当前数据入口。** 使用 `.h5ad`；官方预处理执行计数归一化、`log1p` 和高变基因选择，将高变基因矩阵写入 `.obsm["X_hvg"]`。ST 训练的 TOML 需给出 `[datasets]` 数据目录和 `[training]`；`[zeroshot]` 留出整个细胞类型，`[fewshot]` 留出某细胞类型内的扰动。未明确留出的条目默认参与训练，必须审计划分以防泄漏。需要按实际数据覆盖扰动、细胞类型、批次及对照字段；默认 `gene/cell_type/gem_group/DMSO_TF` 不是 CRISPRi 数据通用规范。不同数据须固定特征集合与顺序。[使用说明](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/README.md)、[默认字段](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/src/state/configs/data/perturbation.yaml)

**本项目只评估小 ST。** `state_sm.yaml` 明确配置：细胞集合长度 128、隐藏维度 672、4 层变换器、8 个注意力头；这是可修改的配置，并非一套无需数据适配的通用预训练模型。小 ST 的完整参数量依赖输入、输出及扰动维数，本次未实例化统计；对应训练显存、时间和最低硬件需求均未知。600M 是另一个可选的状态嵌入模块（SE）；论文“4 节点×8 张 H100”位于 SE 的训练说明，不能移用于小 ST 成本估计，也不应纳入本项目重训计划。[小 ST 配置](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/src/state/configs/model/state_sm.yaml)、[论文第 4.6 节](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full)

**安装与许可。** 官方命令 `uv tool install arc-state`；所查源码版本为 0.11.3，要求 Python `>=3.11,<3.13`、PyTorch `>=2.7.0`，并依赖 `cell-load`、`cell-eval`。代码为 CC BY-NC-SA 4.0；权重及输出另受 Arc 模型许可与可接受使用政策约束。模型许可将商业实体参与或资助的研究排除于其“非商业目的”定义之外，因此不能仅凭“学术项目”认定获得使用授权，产业合作需要单独核实许可范围。[依赖](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/pyproject.toml)、[代码许可](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/LICENSE)、[模型许可第 1.4、4.1 节](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/MODEL_LICENSE.md)、[使用政策](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/MODEL_ACCEPTABLE_USE_POLICY.md)

## 3. Stack：提示驱动的迁移不等于无监督校准

**支持什么。** Stack 根据提示细胞表达谱，将条件信息迁移到查询细胞群，可做表示学习、条件分类及条件表达生成。药物、细胞因子和供体效应有预测验证；遗传扰动表示评估也存在。v2 进一步报告覆盖 892 种药物、细胞因子和遗传扰动的 Perturb Sapiens 预测图谱，不能仍概括为“完全不涉及遗传扰动”。但预测图谱不是全部经过实验验证的图谱，更不是仅输入一个未见基因名称就能生成可靠敲低响应。[v2 摘要及第 2、4.6 节](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v2.full)

**“不需要标签”的准确含义。** 推理接口可以直接使用细胞表达矩阵，无须把扰动名或细胞类型编码作为模型条件；不过提示仍需包含目标条件的信息。官方示例用药物处理后的 T 细胞作为提示，以对照 B/髓系细胞作为查询，并用对照 T 细胞生成合成对照；它不是从完全没有处理响应的细胞中凭空得到药效。示例仍使用 `broad_cell_class`、`sm_name` 来选择细胞和拆分条件；模型后训练也使用带供体、细胞类型或条件信息的数据。[官方预测示例](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/notebooks/tutorial-predict.ipynb)、[v2 第 4.3–4.6 节](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v2.full)

**对无 Th2 扰动数据的判断。** 如有其他环境中相关干预的实测响应，可尝试提示迁移，但只能称为待验证外推。没有独立 Th2 干预真值，就不能证明偏差得到纠正、误差界可信或预测概率已校准；“生成合成对照”也不能替代此类验证。论文自己指出稀有细胞类型及弱扰动的校准仍待建立。上述项目判断是依据任务与验证条件作出的推论，不是论文已验证的 Th2 能力。[v2 讨论](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v2.full)

**输入与配置。** 官方示例下载 `Stack-Large-Aligned` 检查点及对应基因列表，使用 `--base-adata`、`--test-adata`、`--genelist`、`--split-column`。基因对齐代码优先取存在的 `.raw.X/.raw.var`，否则取 `.X/.var`，按指定基因列表对齐并将缺失基因补零。因此须检查 `.raw` 实际保存的内容、基因符号与预处理尺度，不能把 `.raw` 名称当作必然含原始计数，也不能把 SE 或 PCA 向量当作可直接替换的输入。[示例](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/notebooks/tutorial-predict.ipynb)、[生成与对齐代码](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/src/stack/cli/generation.py)

**规模与资源。** v1 表 3 报告模型系列约 69.1M–629M 参数，Large 为 **217M**；此数明确对应该论文配置。v2 仍报告单张 H100 80GB、320GB 系统内存用于预训练，后训练使用单张 H100 80GB、400GB 系统内存；“预训练 2–3 天”是作者在其设置中的报告，不能换算成本项目运行时间。小批次推理的最低显存和内存未从这些数字得到证明。[v1 表 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12803207/)、[v2 第 4.3 节](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v2.full)

**安装与许可。** `pip install arc-stack`；所查包版本为 0.1.3，元数据允许 Python `>=3.9`，官方测试环境为 Ubuntu 22.04、Python 3.10.18、PyTorch 2.5.1+cu121、H100 80GB。存在许可元数据冲突：`setup.cfg` 写 Apache-2.0，但根目录 LICENSE 和 README 写 **CC BY-NC-SA 4.0**；不能据包元数据宣称 Apache 许可。权重及输出另受 Arc 非商业模型许可与使用政策约束，商业资助研究限制与 STATE 类似。[安装说明](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/README.md)、[包元数据](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/setup.cfg)、[LICENSE](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/LICENSE)、[模型许可](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/MODEL_LICENSE.md)、[使用政策](https://github.com/ArcInstitute/stack/blob/cacc2e4b09435c3e536d46237d10b50f222dd144/MODEL_ACCEPTABLE_USE_POLICY.md)

## 4. CellFlow：可编码时间和组合，但需要与任务匹配的监督

**任务与泛化。** 原文覆盖药物、细胞因子、遗传干预、胚胎发育和类器官培养方案；既有已见干预在新环境或新组合中的预测，也测试了完全未见的基因敲除。后者依赖 ESM2 等外部表示，论文明确指出完全未见基因或细胞因子的结果较不稳定。若采用独热编码，只能表示已知干预的新组合，不能据此推断未见的单个干预；不能把基因敲除、激活或其他遗传干预的成绩直接承诺为 Th2 CRISPRi 成绩。[论文图 3、补图 7 及讨论、方法](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full)

**不是只有单终点，但也不是序贯 CRISPRi 成品。** 当前连续发育教程将最早时间点的对照群体映射到其他年龄及扰动条件，用 `logtime` 作协变量，并在实测时间点之间插值；这是横断面分布映射，不是同一细胞的实测纵向跟踪。类器官实验进一步把每种处理的起止日、通路和激活/抑制方向编码为方案，预测方案结果。可保留这些真实时间信息，不能笼统否认它支持时间条件；但流匹配的积分时间本身不等于实验时间，培养方案预测也不证明遗传干预顺序、持效和洗脱效应已学会。[连续发育教程](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/docs/notebooks/201_zebrafish_continuous.ipynb)、[论文图 6 及方法](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full)

**顺序需要显式表示。** CellFlow 聚合干预集合时采用置换不变结构。只给 `{A,B}` 会丢失顺序；若将各自起止时间与干预绑定，才可能表示不同时间方案。进一步用于序贯 CRISPRi 必须另行设计时间/历史特征并获得匹配监督，不能把单次模型连续调用两次当作已验证的 A→B 预测。[条件编码方法](https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1.full)、[条件集合编码源码](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/src/cellflow/networks/_set_encoders.py)

**当前输入要求。** `CellFlow(adata)` 后调用 `prepare_data`：`sample_rep` 指向 `.X` 或 `.obsm` 的细胞表示；`control_key` 是 `.obs` 中布尔对照列；`perturbation_covariates` 指定干预列组；类别的外部表示由 `perturbation_covariate_reps` 指向 `.uns` 字典。可另设 `sample_covariates`、`split_covariates` 和最大组合长度。训练需要来源与目标分布以及条件信息；PCA/潜空间生成后仍需匹配的解码或逆变换，不会自动产生有实验依据的新条件标签。[数据准备及预测 API 源码](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/src/cellflow/model/_cellflow.py)

**规模、安装与许可。** CellFlow 是按任务配置的框架，本次没有核实统一参数量或统一训练 GPU/时长要求，记为未知。官方胚胎教程确实注明运行于 **A100 80GB＋500GB CPU 内存**；它是该大型教程的运行配置，不是所有 CellFlow 任务的最低需求。安装命令为 `pip install cellflow-tools`，当前源码要求 Python `>=3.11`，使用 JAX/Flax、Diffrax、OTT-JAX；GPU 后端须另按 JAX 官方要求配置，装包不代表 CUDA 已可用。[教程资源说明](https://cellflow.readthedocs.io/en/latest/notebooks/200_zebrafish.html)、[依赖](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/pyproject.toml)、[安装文档](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/docs/installation.rst)、[JAX 官方安装](https://docs.jax.dev/en/latest/installation.html)

**当前许可存在实质不一致，不能默认为 MIT。** 同一提交根目录 `LICENSE` 为 MIT，而 `pyproject.toml` 的 `license` 为 `PolyForm-Noncommercial-1.0.0`。这里只记录冲突，不擅自裁定哪一个覆盖所有材料；实际使用应固定版本并向维护者澄清，尤其是商业用途。预印本自身的文章许可也不能替代软件或数据许可。[LICENSE](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/LICENSE)、[包许可字段](https://github.com/theislab/CellFlow/blob/446ed6073c60ac2e8db13c4ea096a43cdec288b2/pyproject.toml)

## 5. 本项目资源约束与建议措辞

以下环境信息来自主任务的服务器只读审计：`nvidia-smi` 显示 `4080 SUPER, 32760 MiB`，按约 **32 GiB 可见显存**记录；虚拟化标签不足以推断实体显卡型号或标准规格。`cgroup memory.max=66571993088`，即 **62 GiB** 容器内存上限；不采用 `free` 所见宿主机约 503 GiB。数据盘早先已满，但主任务 19:50 复查已恢复约396GiB可用空间（见 `server_audit.json`）。目前未确认用户具有 Th2 扰动数据。

据此作出项目判断：

1. 当前空间已恢复，可据实际输入、缓存和输出预算安排试验；运行前仍要重查可用空间，并避免一次下载全库。本次没有删除数据、腾盘或安装训练环境。
2. STATE 只考虑以高变基因输入的小 ST；其配置提供缩小模型与细胞集合的入口，但约 32 GiB 显存、62 GiB 内存下是否可用仍需后续小规模实测，不能凭参数名称保证。[小 ST 配置](https://github.com/ArcInstitute/state/blob/9bbfe78a434a55205e4de834e1ea99f85f7a3add/src/state/configs/model/state_sm.yaml)
3. Stack 原规模训练及 CellFlow 大型胚胎教程所报资源均超出当前容器内存；这不排除缩小任务后的推理或训练，但本次没有做容量验证。[Stack 训练设置](https://www.biorxiv.org/content/10.64898/2026.01.09.698608v2.full)、[CellFlow 大型教程](https://cellflow.readthedocs.io/en/latest/notebooks/200_zebrafish.html)
4. 无 Th2 扰动真值时，三者最多承担候选排序、方法基线或待验证外推；不能写“已完成 Th2 校准”“准确预测序贯 CRISPRi”或用预测替代顺序实验。这里的“校准”特指经独立目标环境真值验证的误差/偏差或概率可靠性，而非简单归一化或条件迁移。

可直接用于方案的表述：**“拟在存储条件满足后，优先评估 STATE 小型 ST 的单终点扰动预测；Stack 用于基于已有表达响应的跨环境假设生成；CellFlow 用于探索带时间信息的条件分布建模。三者在 Th2 序贯 CRISPRi 上的适用性、误差与顺序效应均需专门实验验证，当前不作为已解决能力。”**

证据限制：部分论文网页访问不稳定，本次通过官方页面正文、作者原始代码和平台 API 交叉核查；没有来源支持的硬件最低要求、训练时长、参数量或任务能力均不补写。只核查模型自身，不重复调查数据集原始论文。许可条款按原文记录，不替代正式法律意见。
