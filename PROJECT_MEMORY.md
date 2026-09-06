# perturb Project Memory

## Last verified

- 2026-09-06T20:24+08:00

## Objective

- VERIFIED: 利用原代人 CD4 T 细胞全基因组 Perturb-seq，评估多个单步扰动执行器的可组合性，并构建不确定性感知的短程序贯扰动规划器；最终目标是提出并实验验证 Th2→Th17 的顺序性 CRISPR 干扰方案。
- VERIFIED: 当前阶段已建立隔离模型环境、验证模型/权重可加载、补全数据审计与运行记录；未启动全量模型训练。

## Current state

- VERIFIED: 2026-09-06 D2正式2,000基因面板已在三条件允许训练组合和NTC细胞上完成全量流式计算；共5,384,109个细胞、18,129个非PuroR实测基因，采用CP10K→log1p、20个均值箱和Seurat兼容离散度排序。正式原始HVG与最终面板顺序哈希均为`810bed53174adcb51c80b39f00f28a4ac9031ae0e0b1ef7a47a4f7c33b7b1e99`，14个身份锚点均未触发强制替换；GATA3原始名次2073、IL23R原始名次2835，仍按已冻结资格规则不强制纳入。正式产物为`research/state_d2/d2_hvg_raw.json`和`research/state_d2/d2_gene_panel_2000.json`。
- VERIFIED: 2026-09-06基于正式面板重跑真实D2 Rest试点。固定八个调控靶点（TBX21、GATA3、RORC、STAT4、STAT6、STAT3、BATF、IRF4），集合大小32，输出严格为32×2000，损失4.4689903259且有限，GPU前向、检查点保存和重载均通过；产物为`research/state_d2/d2_state_pilot_contract.json`。IFNG仍因每条件合格细胞过少不进入稳定试点。
- VERIFIED: 2026-09-06官方Replogle检查点迁移审计已按参数语义完成。93/94个目标参数可按键和形状复制，正式面板与官方基因列表重叠220个；官方检查点没有可核实的有序扰动名称，因此扰动投影重叠为0并保持随机初始化，未按整数索引复制。`semantic_assertions_pass=true`，产物为`research/state_d2/transfer_report.json`。
- VERIFIED: 2026-09-06 Scratch/Transfer三种种子的公平训练合同已按同一面板、词表、划分和40,000步预算冻结，批大小固定为64，验证指标为validation MMD；产物为`research/state_d2/training_contracts.json`。数据冻结交叉校验通过，冻结哈希为`e11fbe4c2375878a1058868c4e4bb221495a51036eece81a9582a88a057f0e1c`，产物为`research/state_d2/d2_freeze_validation.json`。
- VERIFIED: 2026-09-06正式训练合同已升级为v2，改为锁定真实官方适配器：2,000基因、12,171扰动、集合大小32、隐藏维328、8层Transformer、12头、批大小64、头部学习率0.001、骨干学习率0.0002和Transfer前1,000步冻结阶段；正式模型配置哈希为`a182832b1d3ab5f3e8494e55e61f7c572dfa97812fd5be26f84342ec6bf3469a`。运行入口必须以`--contract-file`逐字段核验合同。
- VERIFIED: 2026-09-06训练循环已区分Scratch与Transfer：Scratch从第1步训练全部随机参数，Transfer仅在前1,000步冻结Transformer；新增原子`last.ckpt`、优化器/早停/随机数状态保存、合同一致性恢复和`--resume`。远端`e3_state`直接断言确认Scratch骨干更新、Transfer第一阶段骨干不变，且可从第2步恢复到第3步。
- PENDING: 尚未启动Scratch/Transfer的全量40,000步训练、简单基线对比和D2响应评价；当前结果不能给出Transfer或Scratch模型结论。`train_two_phase`与`D2BatchStream`已提供公平两阶段训练接口，不能把试点损失当作性能指标。
- VERIFIED: 2026-09-06补齐正式HVG检测率审计。训练部分可用细胞数为Rest 1,727,344、Stim8hr 1,843,234、Stim48hr 1,813,531，总数与正式HVG的5,384,109一致；每个HVG统计含Ensembl ID和三条件检测率。最终面板的身份覆盖为Naive 5/5、Th1 3/3、Th2 2/3（缺GATA3，原始名次2073）、Th17 2/3（缺IL23R，原始名次2835）；两者因原始名次未差于5000而不触发强制替换，面板强制数为0。
- VERIFIED: 2026-09-06新增`D2BatchStream`和`d2-train`入口。流按冻结的基因×背景划分抽取同条件NTC群体背景和扰动响应集合，支持Scratch/Transfer相同批大小64及两阶段优化；Transfer在官方检查点缺少可核实扰动名称时保持扰动投影随机初始化。该入口已完成编译和命令帮助检查，尚未实际启动40,000步全量作业。
- VERIFIED: 2026-09-06在RTX 4080 SUPER上分别完成Scratch与Transfer真实D2训练流干跑；两者均构建并检查64×32×2000表达/目标张量和64×32×12,171扰动张量，全部有限。训练记录为23,609个组合、验证记录1,822个，面板、词表和划分哈希与冻结合同一致；Transfer干跑确认语义迁移已应用。该干跑不更新权重，也不构成模型性能结论。
- VERIFIED: 2026-09-06远端e3_pipeline测试为`37 passed, 4 warnings`，本地最小环境为`34 passed, 3 skipped`；本地回退近邻实现已消除对scikit-learn的硬依赖。`compileall`和`git diff --check`通过。

- VERIFIED: Git仓库已安全初始化并推送；远端为 `https://github.com/chu62928-cloud/perturb.git`，基线提交 `3cf0df4`，当前唯一大目标分支为 `d2-state-single-step`。`connect_server.py`、`remote_config.json`、原始数据、模型权重、缓存和大型生成物已由 `.gitignore` 排除。
- VERIFIED: 已在 `d2-state-single-step` 提交 `419821d` 冻结 D2 谱系评分器配置、评分函数、D2 元数据审计、固定 guide 校订词表、基因×背景划分和 STATE 模型契约；随后提交 `f4de8ed` 完成远端 D2 产物同步。远端 `e3_pipeline` 测试为 `33 passed, 4 warnings`。
- VERIFIED: 2026-09-06 D2 元数据审计已完成并下载至 `research/state_d2/d2_data_audit.json`。三份文件均为 18,130 维 `_CSRDataset`，形状分别为 2,940,194×18,130、3,032,848×18,130、2,863,571×18,130；原始 X 抽样非负、有限且整数样，D2 完整 CSR 通过摘要作为外部证据。Stim48hr 的 PuroR 位于首列，其余条件位于末列；基因集合相同但列顺序不同，正式读取按 Ensembl ID 建立逐文件列映射，未修改 `.h5ad`。
- VERIFIED: 2026-09-06 使用固定提交 guide 库完成 D2 词表，`unique_observed_guides=24,259`、`eligible_target_count=12,170`，词表哈希 `5c2c29e7f512f0cdf9fdf2ef31a3afeee8b286c76385f622bfe5e6f9065e7cc5`；`NTC=0`，其余扰动按校订目标名称排序，绝不按整数 index 迁移参数。基因×背景划分哈希为 `6debd4dbad36bccc22e47041a1c8461710ecfe5d715d0599740be4f8b809e84b`。
- VERIFIED: 已用通过的 D2 CSR 汇总 `c7c2f4f7116a625c690e5370b8fcc9ceabac0a3b3ad2eb385b9c4a9a66eee7c5` 激活 `development_D2_v2` 的 D2 角色清单；角色哈希为 `c1be44d5f8c7fe98e756c869b4a17e3e4cd8224a3c9ddb9b693c1a91020054bb`。D2 可用于特征选择、划分、效应和阈值选择，D1 仅一次性二级确认，D3/D4 仅最终外部测试。
- PRECHECK ONLY: 已用每条件 10,000 个确定性抽样细胞完成流式 CP10K→log1p、20 个均值箱和 Seurat 兼容离散度排序，产物暂不作为正式 2,000 基因面板；正式面板必须在全体允许训练/NTC 细胞上重跑并通过身份锚点替换闸门。
- VERIFIED PILOT: 在旧的预审计面板上完成一次真实 D2 Rest 试点（TBX21 对 NTC，集合大小 S=32）：8 个候选扰动均完成单 sgRNA/低质量过滤与名称映射检查，输出为 32×2,000，损失有限，GPU 前向、检查点保存和重载均通过。该结果只证明数据读取器和模型接口，待正式全量面板生成后必须按同一合同重跑，不能作为模型性能结论。
- VERIFIED: 已下载 GSE135390 的公开标准化计数文件（SHA-256 `014efb95ba8a810758291d0b5cb34ae0713889bff31287ed3c99327bc266f77d`），按三位供者的 Naive/Th1/Th2/Th17 样本运行冻结评分器；四个身份程序各供者 AUC 均为 1.0，宏平均 AUC 为 1.0，满足至少 2/3 供者方向一致和宏平均 AUC≥0.80。Cano-Gamez 研究已作为独立参考来源记录，待其可公开获取的计数表进入同一 NPZ 接口后补充第二份数值验证；在此之前不把“两个外部数据集均已通过”写入结论。
- VERIFIED: `config/programs.json` 中 Naive/Th1/Th2/Th17 及安全程序尚未冻结；当前 Th2、Th17 等列表为空，因此此前不能声称核心评分基因已经被2,000基因面板覆盖。
- VERIFIED: 2026-09-06 只读核对确认，预注册审计用的36个 Naive/Th1/Th2/Th17 候选基因全部存在于 D2 三个条件相同的18,130维测量轴中。
- VERIFIED: D2均匀分块抽样的HVG预审计纳入17,033个基础质控合格、单sgRNA细胞；按 CP10K、log1p、Scanpy 1.11.5 `seurat` top 2,000计算，Naive覆盖8/8、Th1覆盖5/6（缺STAT4）、Th2覆盖5/9（缺GATA3、CCR4、STAT6、IL4R）、Th17覆盖13/13。该结果仅用于风险预审计，不是冻结模型面板。
- VERIFIED: 本地方案 `CD4_T细胞序贯扰动项目方案.md` 为有效 UTF-8，共 460 行、14 个主要章节，无 TODO/TBD 等占位符。
- VERIFIED: 2026-09-05 元数据盘点显示远端数据目录现有 11/12 个预期 `.h5ad`；11 个均能以 AnnData `backed="r"` 打开并完成首/中/尾探针读取。尚缺 `D3_Rest`。
- VERIFIED: `D2_Stim8hr.assigned_guide.h5ad` 已从旧截断状态恢复至 172,796,432,870 字节。
- VERIFIED: 历史缺失文件总量曾为 5 个；D1_Rest、D3_Stim48hr、D4_Rest、D4_Stim8hr、D4_Stim48hr 已补齐后，目前仅缺 D3_Rest。数据盘约余 323 GiB，仍不能继续无计划全量下载。
- VERIFIED: GPU 报告为 NVIDIA GeForce RTX 4080 SUPER、32760 MiB；容器内存上限 66,571,993,088 字节（约 62 GiB）。
- VERIFIED: 现有 `e3-th-actuator` 为 Python 3.10.20，含 Scanpy 1.11.5、AnnData 0.11.4 和 `pert2state-model` 0.0.1，不含 PyTorch。
- VERIFIED: `e3_state`、`e3_stack`、`e3_primeflow`、`e3_gears` 已建立并通过依赖检查；`e3-th-actuator` 已修复 boto3/botocore 冲突并通过依赖检查。
- VERIFIED: e3_stack 中继承环境的 celloracle/louvain/pygpcca/cellrank 冲突包已移除，STACK 冒烟复测通过且 `pip check` 无未解决项。
- VERIFIED: 已建立独立 `e3_pipeline`（Python 3.11.16，Snakemake 8.30.0，AnnData 0.12.19，Scanpy 1.11.5），不含深度模型依赖。
- VERIFIED: 已部署 Snakemake + Python 流水线骨架，包含数据注册、试点基因选择、基线、评价、支持度和两步束搜索接口。
- VERIFIED: 已用 D1 三个完整条件生成 4,661 个候选；历史 96 基因结构试点保留，新版已拆为响应盲 `primary_64` 与压力测试 `challenge_32`。当前仅有 473 个候选具备抽样 NTC 表达分箱，效应/功能/guide 一致性注释仍待库表。

## Verified results and evidence

- VERIFIED: 具体执行规划已保存为 `research/state_d2/STATE_D2_EXECUTION_PLAN.md`。2026-09-06远端锁定环境复测为`31 passed, 4 warnings`；本地Python 3.14在设置`PYTHONPATH=src`后为26通过、2个因缺少scikit-learn失败、3个因缺少anndata跳过，属于本地依赖不完整而非本轮文档变更回归。`compileall`、新增JSON解析和项目记忆检查均通过。
- VERIFIED: 谱系标记文献证据已写入 `research/state_d2/th_lineage_marker_evidence.md`；D2测量轴和抽样HVG覆盖报告为 `research/state_d2/d2_hvg_marker_precheck.json`。两者明确区分身份锚点、条件依赖效应因子及激活/应激/凋亡混杂程序。
- VERIFIED: D2三个条件中，基础质控合格且标记为单sgRNA的关键调控因子扰动均有可用细胞：TBX21每条件297–331、GATA3为240–387、RORC为162–175、STAT4为104–139、STAT6为248–327；IFNG仅0–6，不适合作为稳定扰动验证靶点。以上为obs标签计数，不等同于通过后续敲低效率过滤。
- VERIFIED: 本地旧数据审计位于 `research/server_audit.json`；其中 D2 截断结论已过时，需由本轮新审计替代。
- VERIFIED: `/etc/network_turbo` 可读；在临时 shell 中启用后，GitHub 和 Hugging Face 均返回 HTTP 200。
- VERIFIED: 2026-09-05 远端已有完整三条件供者 D1、D2；D3 已有 Stim8hr/Stim48hr，D4 已有 Rest/Stim8hr/Stim48hr，但 D4_Stim48hr 的 CSR 审计失败；D3_Rest 尚缺。
- VERIFIED: STATE 官方 `ST-HVG-Replogle` 检查点已在 RTX 4080 SUPER 上加载并前向：输入 `[1,64,2000]`，输出 `[64,2000]`，约49.4M参数，峰值显存约214 MB。
- VERIFIED: STACK 官方 `Stack-Large-Aligned` 检查点已在 RTX 4080 SUPER 上加载并前向：输入 `[1,8,15012]`，输出 `[1,8,15012]`，细胞嵌入 `[1,8,1600]`，约217.8M参数，峰值显存约3.53 GB。
- VERIFIED: PRiMeFlow 缩小 `DynamicsMLP` 已在 GPU 前向，并保存/新实例回载检查点；输入输出 `[4,8]`，回载最大绝对误差为0。
- VERIFIED: GEARS `cell-gears==0.1.2` 与 `torch-geometric==2.6.1` 已在 GPU 完成单基因扰动索引预测，并保存/回载检查点；输入 `[8,1]`，输出 `[1,8]`，回载最大绝对误差为0。
- VERIFIED: `pert2state-model==0.0.1` 已完成 6×3 小矩阵拟合和预测，输出有限值。
- VERIFIED: STACK、PRiMeFlow、GEARS、pert2state 的日志位于远端 `CRISPR_perturb_runtime/smoke/`；本地副本位于 `research/smoke_logs/`。环境锁定位于 `research/runtime_manifests/`。
- VERIFIED: 旧注册表 `CRISPR_perturb_runtime/pipeline/metadata/dataset_registry.json` 仍保留 7/12 的历史登记；`metadata/external_audits_20260904/registry_refresh/metadata/dataset_registry.json` 是此前 10/12 的独立快照；本次新建且不覆盖旧文件的 `metadata/external_audits_20260905/registry_refresh/metadata/dataset_registry.json` 登记 11/12 个现存文件，并更新了 D2_Stim8hr、D4_Stim48hr 的大小、修改时间和指纹。
- VERIFIED: 已在固定 GitHub 提交 `aa5c84a973c0e1a090b0072dc5b080bf7fbbed38` 下载并审计作者的 guide 库、guide 效率、DE 摘要和 Th1/Th2 外部参考；当前 DE/效率摘要仅作审计，未用于 D2 调参。
- VERIFIED: 本轮已将目标角色协议改为 D2 开发供者、D1 一次性二级锁定确认供者、D3/D4 最终外部测试供者；新增不可覆盖的 `donor_roles.v1` 清单接口、D2 专用运行目录和角色哈希字段。2026-09-05 D2 三条件完整 CSR 闸门已通过，但角色仍未激活。
- VERIFIED: 2026-09-02 对 D2 三份文件执行了覆盖 `X/indptr`、`X/indices`、`X/data` 全部逻辑元素的只读分块审计（每块 1,024 行，支持断点续跑）。D2_Rest 与 D2_Stim48hr 全部通过；D2_Stim8hr 在第 751 块（行 `769024:770048`，原始非零范围 `3614727211:3619483113`）触发 HDF5 `wrong B-tree signature`。
- BLOCKED (历史): 2026-09-02 D2 三文件完整 CSR 审计曾因旧 D2_Stim8hr 触发 B-tree 错误而汇总为 `BLOCKED_D2_CSR`；该摘要哈希为 `2897fb7fe8c6d1e35be7e3ba5024601a2e888bd5f96933450a147fe4a61908ba`。2026-09-05 新文件复审已替代该结论。
- VERIFIED: 2026-09-05 D2 三条件完整 CSR 审计汇总为 `PASS`，摘要哈希 `c7c2f4f7116a625c690e5370b8fcc9ceabac0a3b3ad2eb385b9c4a9a66eee7c5`；D2_Stim8hr 全部 2,962 个块重新读取，`indices=data=14,352,017,864`，审计前后大小、修改时间和采样指纹稳定。D2 仍未激活角色、未读取生物学响应、未生成效应矩阵或训练模型。
- VERIFIED: D2_Rest 的 CSR 长度为 `indptr=2,940,195`、`indices=data=11,668,960,770`；D2_Stim48hr 为 `indptr=2,863,572`、`indices=data=11,417,773,379`。两份审计前后文件大小、修改时间和首尾指纹一致；D2_Stim8hr 失败前后指纹也一致。远端已写入不可覆盖的 `donor_roles_v1.pending.json` 与 `donor_roles_v1.blocked.json`，没有生成 active 清单。
- VERIFIED: Snakemake `preflight → audit → prepare_pilot → freeze_data` 实际运行完成；`e3_pipeline` 的 13 个核心测试通过，`pip check` 无未解决依赖；新增基线、连续状态、双闸门和几何匹配/封存/评分命令级 smoke 通过。
- VERIFIED: 已生成不覆盖历史结果的 `pilot_64_primary_v1.json`、`pilot_32_challenge_v1.json`、`pilot_manifest_v1.json` 和 `data_contract_frozen_v1.json`；数据契约当前标记 `DATA_CONTRACT_READY`（尚未升级 `DATA_VALID`），且明确 `d2_responses_used=false`。
- VERIFIED: 2026-09-02 已从官方对象复用并登记两个预先核实的 D1_Rest `X/indptr` HTTP Range 补丁块（每块 1001 个小端 int64、8008 字节），未覆盖原始 `.h5ad`。补丁文件为 `metadata/patches/d1_rest_x_indptr_450450_451450.bin`（SHA-256 `936e7bfaa2fa61ca2b6488af16961b01a6efc5b89928838d42a761bba4c9dd10`）和 `metadata/patches/d1_rest_x_indptr_451451_452451.bin`（SHA-256 `58289bdd31396760fd75b2a1c0d5801c9fe7bda548e4486babd0a3a3815e7a3b`）；sidecar `metadata/d1_rest_csr_patch_v1.json` artifact hash `7fc7f844254e661a2b1ad4833ce7e513be7d376f66ff46058b69717d0b7de745`，`raw_input_modified=false`、`d2_responses_used=false`。
- VERIFIED: 同日 `d1_csr_integrity.v2` 将 D1_Rest 区分为 `raw_valid=false`（下降 1、内部零指针 2002）和 `patched_pointer_valid=true`；D1 Stim8hr/Stim48hr raw/patched 均 true，综合 `raw_all_valid=false`、`patched_all_valid=true`。该审计明确范围为 `X/indptr_structure_only`，不等同于完整 CSR 值数组可读。
- BLOCKED: 补丁边界验证读取 D1_Rest 恢复行的原始 `X/indices` 时，在 row 450448、数据切片 `1726369350:1726374503` 触发 HDF5 `OSError: Can't synchronously read data (wrong B-tree signature)`；独立探针还显示 `X/data` 在约 `1734395950` 起同样失败。因此不能证明其余 CSR 数组可信，未启动三条件 effect matrix。远端 `metadata/d1_rest_csr_patch_verify_v1.json` 为 `status=BLOCKED`、`all_rows_valid=false`。
- VERIFIED: 补丁 overlay、坏 hash/范围拒绝、raw fail-closed、D2 guard 测试及既有测试共 21 项通过；远端 `e3_pipeline` compileall 通过，`pip check` 为 `No broken requirements found`。证据驱动契约 `metadata/data_contract_frozen_v7.json` 保持 `DATA_CONTRACT_READY`，明确记录 effect matrix 缺失、补丁边界验证阻塞和外部补丁尚未获得科学口径显式接受。
- VERIFIED: 角色阻塞后尝试运行 `activate-roles` 被明确拒绝，active 清单不存在；最新远端测试为 `31 passed, 4 warnings`，`compileall` 通过，`pip check` 无未解决依赖。CSR 测试已覆盖正常读取、断点恢复、指针下降、长度不一致、索引越界、非有限值和源文件变化拒绝；基线接口已增加固定 `pert2state` 适配器输出的同划分登记与有限值校验，并新增仅基于 D2 训练/验证集的 STATE zero-shot/轻量适配决策模块。
- VERIFIED: 对 D2_Stim8hr 失败范围执行独立 h5py 默认驱动与 `sec2` 驱动重试，`X/indices` 和 `X/data` 均再次报 `wrong B-tree signature`；该阻塞不是单一读取驱动或短暂缓存问题。
- VERIFIED: 2026-09-05 新 D4_Stim48hr 完整 CSR 审计在第 2,021 块（行 `2069504:2070528`，非零范围 `9774408325:9779444902`）发现 1,537 个越界索引；矩阵形状为 `[2815784,18130]`，首个异常绝对 `indices` 位置 `9778743625`，值 `4575657221408425920`。独立探针确认 `data` 仍有限且非负、源文件前后指纹稳定；该文件判定 `BLOCKED_D2_CSR`，不得用于外部测试。
- VERIFIED: 2026-09-04 独立外部文件完整 CSR 审计已完成：新 D1_Rest、D3_Stim8hr、D3_Stim48hr、D4_Rest、D4_Stim8hr 五份文件均 `PASS` 且 `full_csr_read_valid=true`。审计只读原始 HDF5，不改变 D2 角色协议或读取外部测试响应。
- VERIFIED: 新 D1_Rest 大小 `142,828,875,662`、采样指纹 `61733c640f7d46d9bf777f26bd3b18314edc8cdf5bfdf4ad6bebe0fdf2cdc78d` 与刷新登记快照一致；审计前后大小、修改时间和采样指纹均未变化。第一次整组 D1 命令因旧注册表修改时间过期而 fail-closed，不能作为内容失败结论。
- VERIFIED: 2026-09-05 仅做元数据级复核时，远端已存在的 11/12 个 D1–D4 文件均能以只读 AnnData 打开，登记文件大小一致；缺少 `D3_Rest.assigned_guide.h5ad`。该复核不替代完整 CSR 审计。
- VERIFIED: 2026-09-05 在刷新登记快照（不覆盖旧注册表）下，新 D2_Stim8hr 全量 CSR 读取通过；新 D4_Stim48hr 在 `X/indices` 越界处阻塞。D1_Rest、D3 两文件和 D4_Rest/D4_Stim8hr 的 2026-09-04 通过报告保持不变。

## Decisions and invariants

- PENDING: 建议先冻结14个谱系身份锚点（Naive: TCF7/LEF1/CCR7/SELL/MAL；Th1: TBX21/IL12RB2/CXCR3；Th2: GATA3/CCR4/PTGDR2；Th17: RORC/CCR6/IL23R）。正式训练面板仅对“D2测得、映射无歧义、训练部分表达达标但未进top 2,000”的身份锚点执行少量替换；效应细胞因子和STAT总表达不自动强制加入。该规则须在正式HVG计算前冻结。
- VERIFIED: D2的Rest、Stim8hr、Stim48hr是刺激背景，不是Naive、Th1、Th2、Th17标签。独立极化参考可用于先验验收评分器；D2内部只能评价扰动是否沿冻结谱系程序移动，不能据此宣称完成真实谱系转换。
- VERIFIED: GitHub/Hugging Face 下载必须在临时子 shell 内先执行 `source /etc/network_turbo`，不得把代理写入持久配置。
- VERIFIED: 不删除已有原始数据；磁盘至少保留 220 GiB，低于阈值时停止新增大型下载。
- VERIFIED: STATE/STACK 官方权重只按非商业学术研究许可使用；项目记录不得保存访问令牌。
- VERIFIED: PRiMeFlow 当前没有公开预训练权重，只验收缩小模型与自建测试检查点，不能称为官方预训练模型。
- VERIFIED: e3_state 使用 Python 3.12.4 + PyTorch 2.7.0 CUDA 12.8；e3_stack/e3_gears 使用 Python 3.10.20 + PyTorch 2.11.0 或 2.7.0 CUDA 12.8；e3_primeflow 使用 Python 3.12.4 + PyTorch 2.7.0 CUDA 12.8。以上组合均已实测 GPU 可用，不强行安装会长时间阻塞的官方 CUDA 12.1轮。
- VERIFIED: 初始研究划分默认 D1 为开发供者、D2 为锁定外部供者；在正式冻结清单生成前不得使用 D2 响应调参。
- VERIFIED: 旧 D1 产物、旧 D1 损坏审计与两条件 D1 读取 smoke 均保留为历史只读证据，不覆盖、不改名冒充 D2 结果；D1 smoke 只读了元数据或局部刺激条件，未用于模型训练、调参或模型选择。
- VERIFIED: 环境加载成功不代表模型已适用于 Th2→Th17 或序贯 CRISPR 干扰。
- VERIFIED: 新编排环境统一采用已验证的 AnnData 0.12.19；大型 backed CSR 读取只使用 slice-first 路径，避免 AnnData/Scipy 组合的高级索引兼容性问题。
- VERIFIED: 96 基因当前为“结构试点”；在 guide 库映射、效应强弱、功能类别和刺激依赖注释补齐前，不得将其作为最终分层试验集合。
- VERIFIED: 修订后主要统计只使用响应盲 `primary_64`；`challenge_32` 仅作压力测试。连续状态以潜空间近邻区域为主，Leiden 只作报告；D1 自助法不解释为供者泛化。
- VERIFIED: 科学状态链已固定为 `DATA_VALID → MODEL_GLOBAL_VALID → MODEL_STATE_VALID → COMPOSITION_VALID → PLANNER_VALID → EXPERIMENT_VALID`；当前仍处于 DATA_CONTRACT_READY，单步模型未获科学放行。
- VERIFIED: 新增固定提交 guide 库的校订接口（保留原始标签、校订标签、脱靶与冲突原因），以及仅接受 D1 路径的逐 guide/基因级效应矩阵构建接口。
- VERIFIED: 2026-09-02 已实际运行三份 D1 obs 指南校订：23,651 个独特靶向指南中 23,341 个由固定提交库映射，独特指南映射率 0.9868927318；三个条件细胞级映射率分别为 Rest 0.9717934264、Stim8hr 0.9720675193、Stim48hr 0.9711508243。多指南/无指南、低质量、库缺失、库标记和原始标签不一致分开记录；NTC 无靶点映射视为合法。
- VERIFIED: 2026-09-02 已从三个 D1 NTC 条件按固定 log1p(CP10K) 规则生成共享特征顺序：18,117 个共同标准基因中冻结 2,000 个，基因顺序哈希 `dda7a714720a8e9b294d34c502a781b1593f79e44dcc32194d4cf8be6ef2adee`；每条件使用 4,096 个 NTC，69 个可用试点基因强制纳入，27 个缺少共同表达特征并保留为明确兼容性记录。
- VERIFIED: 2026-09-02 已从 D1 NTC 2,000 特征按 CP10K 拟合 50 维中心化随机 PCA，三条件各 10,000 个细胞，共 30,000×50；状态区域使用 sklearn 近邻算法，不构造全距离矩阵。已生成 30 个局部区域、5 折缓冲划分和 gene/state/gene×state 冻结清单；状态哈希 `5e9fd8a3fff962092aabe7be57389d607554f2bee8eac6392dd8d97dcda49329`，综合 split 哈希 `016d2f4d8f5dc9c69b0e22276bb6a42035851b1ce3d7505ce86b89ed85462168`。
- VERIFIED: 已将 effect matrix 读取器改为单次打开 HDF5、CSR 分块读取、分类整数 codes 和原子结果写入；缺少 `total_counts`、固定 gene ID 或损坏 CSR 指针会在进入 SciPy 前明确失败。
- BLOCKED: 远端三份 D1 文件完整性审计发现 `D1_Rest.assigned_guide.h5ad` 的 `X/indptr` 在第 450449 行下降，450450–452451 行有 2002 个内部零指针；独立 raw CSR 读取在 block 880 报 `raw_stop=0`，旧实现随后发生 segmentation fault (exit 139)。该原始文件只读，不能在项目内修复；当前不能生成可信的三条件 D1 ground truth 或升级 DATA_VALID。
- VERIFIED: 2026-09-02 运行 `d1_csr_integrity.v1`：D1 Stim8hr 和 Stim48hr 均通过；D1 Rest `valid=false`、`decreasing_pointer_count=1`、`first_decreasing_row=450449`、`zero_pointer_count_after_start=2002`；证据副本为 `research/pipeline_outputs/d1_csr_integrity_v1.json`。
- VERIFIED: 2026-09-02 已在独立目录 `effect_smoke_stim_only` 完成 D1 Stim8hr/Stim48hr 的真实分块效应读取：共 1,528 条 guide-effect 记录、每条件 2,000 个 gene 向量，全部有限值，`d2_responses_used=false`；这只是读取器稳定性 smoke，不是三条件 ground truth。

## Important files and paths

- 本地方案：`CD4_T细胞序贯扰动项目方案.md`
- 本地详细实施方案：`perturb_implementation_plan.md`
- 本地证据：`research/`
- 本地连续性记录：`PROJECT_MEMORY.md`
- 远端原始数据：`/root/autodl-tmp/CRISPR_perturb/`
- 远端运行根目录：`/root/autodl-tmp/CRISPR_perturb_runtime/`
- 远端流水线：`/root/autodl-tmp/CRISPR_perturb_runtime/pipeline/`
- 本地流水线源码：`src/cd4perturb/`、`workflow/Snakefile`、`config/`、`requirements/e3_pipeline.requirements.in`
- 本地流水线产物：`research/pipeline_outputs/`
- 远端连接入口：`connect_server.py`（包含凭据，不得复制其凭据到日志或项目记忆）

## External systems and background jobs

- VERIFIED: AutoDL 服务器可通过本地 `connect_server.py --cmd '<只读命令>'` 检查；连接配置仅保留在该脚本。
- VERIFIED: effect matrix 后台 PID `167116` 已退出且无产物；前台诊断分别以 segmentation fault (exit 139) 退出，后续完整命令以 exit 1 的输入损坏错误退出；2026-09-06T20:24+08:00当前无运行中的D2训练作业。
- VERIFIED: 独立两条件 smoke PID `173671` 已正常退出；远端产物 `/root/autodl-tmp/CRISPR_perturb_runtime/pipeline/effect_smoke_stim_only/results/d1_effect_matrix_v2.json` 为 140,890,154 bytes，文件 SHA-256 `2b4e0a5a33a475552e723e48f0ee917994e062928233a439d35f5be6163b9b37`、内容 effect_hash `ebca45c13129f0e1be2c4b373261f93ba7a9cc59096bc37578ea616a20497436`；主三条件结果仍不存在。
- VERIFIED: 证据驱动契约复核已执行，远端新文件 `metadata/data_contract_frozen_v4.json`（1,265 bytes，SHA-256 `018d88043e81e978ad5df9ab75dcf7fae4651fe01050d449138dd1dad12dd854`）保持 `DATA_CONTRACT_READY`，原因明确为正式三条件 effect matrix 缺失及 D1 CSR integrity 无效；本地副本位于 `research/pipeline_outputs/data_contract_frozen_v4.json`。
- 安全复查：`df -h /root/autodl-tmp`、`nvidia-smi`、`conda env list`。
- VERIFIED: 外部审计报告已同步至本地 `research/pipeline_outputs/external_audits_20260904/` 和 `research/pipeline_outputs/external_audits_20260905/`；2026-09-05 的 D2/D4 报告与独立 D4 失败探针 `artifact_hash` 均已重算通过。远端审计进程 `5367`、`5644` 已退出，当前无 CSR 审计后台任务。

## Immediate continuation

1. 从冻结词表和训练组合中确定64个技术试点扰动，优先八个机制靶点，其余按至少两个训练背景各有32个细胞及最小细胞数排序补齐；不得读取验证或测试性能。
2. 在轻量128维、4/4层、8头模型上运行200步训练试点；通过损失下降、有限值、形状、显存、名称映射和重载一致性硬断言后，再运行Scratch/Transfer正式328维架构各20步优化冒烟。
3. 冒烟全部通过后，按Scratch/Transfer交替顺序依次运行三个种子的正式训练；所有检查点仅用验证MMD选择，六个最佳检查点冻结前不得评价测试响应。
4. 要求数据提供方修复或重新提供 `D4_Stim48hr.assigned_guide.h5ad`；保持当前原始文件只读，修复后按同一 `audit-csr` 命令复跑。继续保留 D4 为外部测试供者，但该条件在通过前不得使用。
5. 获取缺失的 `D3_Rest.assigned_guide.h5ad` 后，先做登记和完整 CSR 审计，再考虑 D3 三条件外部测试。
6. 用 `state-regions` 冻结连续潜空间区域及缓冲，完成基因、状态、基因×状态三类留出及三层评价空间。
7. 依次评价 `MODEL_GLOBAL_VALID` 与 `MODEL_STATE_VALID`；至少两个执行器通过后，再运行 `match-composition → score-composition` 和支持度闸门。

## Blockers and unknowns

- VERIFIED: STATE 与 STACK 权重均已从官方 Hugging Face 仓库下载并完成实际 GPU 加载；许可文件已保存，未记录令牌。
- VERIFIED: 当前 D1 96 基因结构试点和 4,661 候选已生成；指南校订、共享特征顺序、真实 50 维 NTC 潜空间和三类划分已有证据。
- BLOCKED: effect matrix 完成后的 guide/gene QC 汇总、证据驱动 `DATA_VALID` 验证、简单 baseline、模型双闸门和 naive→Th1 fate evaluator 尚未开始；D4_Stim48hr 的 `X/indices` 越界和 D3_Rest 缺失是当前外部数据阻塞。
- BLOCKED: D4_Stim48hr 在修复或重新下载前不能用于最终外部测试；D2开发角色、数据审计、划分和正式面板已冻结，不受该外部测试阻塞影响。

## Update history

- 2026-09-06T20:24+08:00: 提交并推送`83c4f3c`，修正正式模型合同、Scratch/Transfer阶段语义和可恢复检查点；远端e3_pipeline为38项通过、3项因无PyTorch跳过，e3_state直接优化断言全部通过。重新生成v2合同，正式模型配置哈希为`a182832b1d3ab5f3e8494e55e61f7c572dfa97812fd5be26f84342ec6bf3469a`；尚未启动模型训练。

- 2026-09-06T19:10+08:00: 提交`0339ee5`完善正式HVG映射、检测率审计、D2批流和`d2-train`入口；提交`9c930a5`加强迁移报告目标哈希/重叠断言并加入批流合成数据测试；提交`5a1cd9d`完成真实D2 Scratch/Transfer批流干跑产物；提交`f30c52f`更新连续性记录。最终远端测试为37项通过、1项因e3_pipeline无PyTorch跳过，`pip check`、`compileall`和项目记忆检查通过；所有提交均已推送到唯一分支`d2-state-single-step`。

- 2026-09-06T16:05+08:00: 写入D2专属STATE可执行规划，建立安全Git基线并推送`main`提交`3cf0df4`；创建并推送唯一大目标分支`d2-state-single-step`。远端e3_pipeline复测31项通过、4条已知警告。

- 2026-09-06T15:51+08:00: 完成Th谱系评分基因前置核对。36个候选全部存在于D2三条件测量轴；17,033细胞抽样HVG预审计显示Naive和Th17完全覆盖、Th1缺STAT4、Th2缺GATA3/CCR4/STAT6/IL4R。提出只对14个身份锚点执行表达资格和少量强制纳入审计，并将外部极化评分器验收与D2扰动方向验证分开。

- 2026-09-02T14:50+08:00: 以登记表的 `observed_size`、`mtime`、采样指纹作为审计前置核对并完成复跑。D2_Rest（2,940,194 行）与 D2_Stim48hr（2,863,571 行）通过；D2_Stim8hr 在第 751 个 1,024 行块的 `X/indices`/`X/data` 范围 `3614727211:3619483113` 报 `wrong B-tree signature`。角色意向保持 `PENDING_D2_CSR_AUDIT`，并追加不可覆盖的 `donor_roles_v1.blocked.2897fb7fe8c6.json`，未激活 D2。

- 2026-09-02T13:30+08:00: 新增 D1_Rest 两块固定 `indptr` HTTP Range 补丁与严格 overlay/哈希/范围/D2 guard；指针审计 patched pointer valid，但边界读取暴露 `X/indices`/`X/data` B-tree 签名错误，verify 明确 BLOCKED，未运行三条件 effect matrix，契约保持 DATA_CONTRACT_READY。

- 2026-09-04T16:13+08:00: 完成新 D1_Rest、D3_Stim8hr、D3_Stim48hr、D4_Rest、D4_Stim8hr 的全量 CSR 审计；五份均 PASS，审计前后源文件指纹稳定。发现 D3_Rest 与 D4_Stim48hr 尚缺；旧 D1 两个条件的历史通过报告保留且未重复读取。远端测试 31 项通过，compileall 与 pip check 通过。

- 2026-09-05T19:24+08:00: 新 D2_Stim8hr 全量 CSR 审计通过，D2 三条件汇总 PASS；新 D4_Stim48hr 在第 2,021 块发现 1,537 个越界 `X/indices`，汇总 BLOCKED_D2_CSR。元数据盘点为 11/12，仍缺 D3_Rest；D2 角色未自动激活。

- 2026-09-02T10:45+08:00: 完成并上传 guide-qc、共享 2000 gene order、D1 NTC 50 维 latent、近邻 state regions 与三类 split 实现；远端测试 15 项通过。三条件 D1 effect matrix 已以分块 CSR 方式启动，D2 仍未读取。
- 2026-09-02T11:35+08:00: 定位 effect matrix 真实失败根因：D1 Rest `X/indptr` 内部 2002 行为零且出现一次下降，造成 SciPy 路径 exit 139；新增分类 codes/单句柄/原子写入/CSR integrity guard、19 项远端 pytest 全通过，并生成 `metadata/d1_csr_integrity_v1.json`。在获取完整原始文件前保持 DATA_CONTRACT_READY。
- 2026-09-02T13:35+08:00: 完成两个固定 `X/indptr` 补丁块的 sidecar/overlay、指针审计和边界验证。指针审计为 `raw_all_valid=false`、`patched_pointer_all_valid=true`（scope 仅 `X/indptr_structure_only`）；边界行读取在 `X/indices` 触发 `wrong B-tree signature`，且独立探针在 `X/data` 也触发同类错误。`d1_rest_csr_patch_verify_v1.json` 为 BLOCKED，未启动三条件 effect matrix；远端 21 项 pytest、compileall、pip check 均通过；契约 `data_contract_frozen_v7.json` 保持 DATA_CONTRACT_READY。
- 2026-09-01T22:12+08:00: 落实修订计划：隔离 primary_64/challenge_32，加入连续潜空间区域与缓冲留出、D1 分层 Bootstrap、全局/状态双闸门、封存式组合性匹配和多目标安全契约；远端 `pytest` 13 项通过，Snakemake `freeze_data` 完成。

- 2026-09-01T14:52+08:00: Continuity files initialized.
- 2026-09-01: 写入已核实的方案、数据、资源、网络和环境准备状态。
- 2026-09-01T16:05+08:00: 完成四个执行器与 pert2state 最小验收、模型权重保存、依赖锁定、数据审计副本和方案实施附录；下一步转入小规模基线适配。
- 2026-09-01T16:18+08:00: 清理 STACK 环境残余依赖冲突，更新其环境锁定与冒烟日志；五个环境的最终 `pip check` 均通过。
- 2026-09-01T17:31+08: 建立并上传 `e3_pipeline`，完成 Snakemake `preflight → audit`、D1 候选/96 基因结构试点、非深度核心 smoke 与 4 项测试；发现并修正 AnnData backed CSR 高级索引兼容问题。
- 2026-09-01T17:36+08: 增加三类留出、组合性闸门、模型分歧、基线接口和可恢复的试点数据读取；远端 `pytest` 5 项通过、`pip check` 通过。
- 2026-09-01T17:40+08: Snakemake 新增并强制执行 `prepare_pilot`，完成 D1-only 候选与 96 基因结构试点重跑。
- 2026-09-01T17:42+08: 增加不可覆盖的 D2 锁定发布清单接口；远端 `pytest` 6 项通过，未执行实际 D2 发布。
- 2026-09-01T17:52+08: 在固定 GitHub 提交上下载并审计四份小型公共元数据；记录 guide/效应摘要为审计输入，暂不用于 D2 选择。
- 2026-09-01T17:58+08: 公开元数据审计后复跑流水线测试，远端 `pytest` 6 项通过、`pip check` 通过；`MODEL_VALID` 及后续闸门仍保持 PENDING。
- 2026-09-01T18:01+08: 更新项目方案新增“流水线实施状态”附录，核对方案为 460 行、14 个主要章节。
