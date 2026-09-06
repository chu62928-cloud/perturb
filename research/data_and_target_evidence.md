# 数据、参考状态与资源核查

核查日期：2026-08-31。以下区分远程实测、原作者文档与方案推断。

## 远程实测

可复查原始记录：`server_audit.json`；只读脚本：`audit_server.py`。脚本通过已有连接配置运行，不输出凭据，也不修改源数据。

- D1_Rest：3,074,496 × 18,130；142,828,875,662 字节；HDF5、AnnData 和首/中/尾行读取成功。
- D1_Stim8hr：2,789,727 × 18,130；154,774,003,188 字节；读取成功。
- D1_Stim48hr：2,805,354 × 18,130；156,246,167,022 字节；读取成功。
- D2_Stim8hr：112,828,416,000 字节；HDF5 报告截断，文件内部预期长度和源端长度均为 172,796,432,870 字节。不能用作第二供者。
- D1 三个文件特征集合相同，但 48 小时文件的列顺序不同；必须按基因编号对齐。
- 过滤 `low_quality`、保留单 sgRNA 后，Rest / 8h / 48h 的 NTC 数分别为 76,543 / 69,632 / 73,534，靶向且具有原始基因编号的细胞数为 1,675,494 / 1,537,130 / 1,574,301。尚未做作者靶基因校订、有效性或脱靶过滤。
- 原文件 `targeting single sgRNA` 分组也包含 NTC；必须联合 `guide_type` 判定。
- 样本行总和等于 `total_counts`，抽样值非负且为整数型数值；与作者对 X 为 UMI 计数的定义一致。
- GPU 驱动报告 NVIDIA GeForce RTX 4080 SUPER，32760 MiB；虚拟化环境下不据此认定实体显卡规格。容器内存上限为 66,571,993,088 字节（62 GiB）。
- 早先数据盘接近满；19:50 快照可用 425,340,674,048 字节（约 396 GiB）。没有执行清理或恢复下载。

此前两个 D1 刺激文件已对比源端字节数、首尾各 1 MiB 的 SHA256；一致，但不是全文件密码学一致性证明。新增 Rest 目前做了大小、结构与行抽样检查。

## 原始论文与实验设计

[Zhu、Dann 等，Cell，2026，DOI 10.1016/j.cell.2026.08.002](https://doi.org/10.1016/j.cell.2026.08.002) 的公开摘要确认：约 2200 万原代人 CD4 T 细胞、4 位供者、静息和刺激条件；扰动响应随条件而变，并用扰动特征研究极化等状态。公开记录 [GSE314342](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE314342) 与之对应。

[官方样本表](https://github.com/emdann/GWT_perturbseq_analysis_2025/blob/master/metadata/suppl_tables/sample_metadata.suppl_table.csv) 显示 D1=CE0008162，D2=CE0010866，D3=CE0008678，D4=CE0006864。D1/D2 的 Rest 和 Stim8hr 在 R1，全部 Stim48hr 与 D3/D4 的其他条件在 R2。因此 D1 的 8h/48h 差异还伴随处理批次变化，不能全归因于时间。

[作者数据说明](https://github.com/emdann/GWT_perturbseq_analysis_2025/blob/master/metadata/data_sharing_readme.md) 明确 X 是 UMI 计数、PuroR 是嘌呤霉素抗性标记表达；细胞级原始靶基因标签在事后校订之前。`sgrna_library_metadata.suppl_table.csv` 中的 `target_gene_id` / `target_gene_name` 应作为校订映射，保留原始字段和映射审计。

## 可省去全量下载的数据

公开 S3 清单于本次核查返回以下对象大小；均未在本次任务下载到服务器：

| 对象 | 字节数 | 作用与限制 |
|---|---:|---|
| `GWCD4i.DE_stats.h5ad` | 16,786,240,107 | 全供者汇总效应矩阵，V0 发现用途；不用于声称留供者独立测试 |
| `GWCD4i.pseudobulk_merged.h5ad` | 44,566,657,140 | 分供者/guide 的伪总体计数，可按折重做效应估计 |
| `GWCD4i.DE_stats.by_donors.h5mu` | 16,866,278,447 | 供者对效应；组合间可能共享供者，不能任意当独立训练测试 |
| `GWCD4i.DE_stats.by_guide.h5mu` | 29,424,424,894 | guide 复现性辅助 |
| `suppl_tables/DE_stats.suppl_table.csv` | 7,276,533 | 每个扰动的质量摘要，不是完整扰动×表达基因效应矩阵 |
| `suppl_tables/sgrna_library_metadata.suppl_table.csv` | 9,935,249 | 校订靶基因及脱靶信息 |
| `suppl_tables/guide_kd_efficiency.suppl_table.csv` | 13,319,852 | guide 效率摘要，若包含测试数据只能审计，不能据此挑选测试样本 |
| `suppl_tables/Th1Th2bulkRNAseq_DESeq2_results.csv.gz` | 45,347,130 | 外部极化验证，只覆盖作者挑选的候选 |

对象根地址：[公开 S3 清单](https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/?list-type=2&prefix=marson2025_data/)。代码入口：[作者分析仓库](https://github.com/emdann/GWT_perturbseq_analysis_2025)。

## 可用的 Th2 / Th17 参考

[Cano-Gamez 等，Nature Communications，2020](https://doi.org/10.1038/s41467-020-15543-y) 包含人初始与记忆 CD4 T 细胞的极化研究，单细胞中包含 Th2、Th17 等条件。适合作为公开目标参考；其初始来源、记忆来源与培养条件必须分别处理，不能仅凭培养标签把所有细胞当作纯 Th2 或 Th17，更不是 Th2 经基因干预转成 Th17 的真值。

[Open Targets 项目页](https://opentargets.org/projects/effectorness) 将下载指向 [BioStudies S-BSST2978](https://www.ebi.ac.uk/biostudies/studies/S-BSST2978)。本次读取其公开 API 确认：

- `bulk_RNAseq-20260511T152800Z-3-001.zip`：4,462,172 字节，原始基因×样本计数，适合先构建目标特征。
- `scRNAseq.zip`：8,027,321,152 字节，10X 3′ v2 的 UMI 数据，供后续分布目标使用。
- `Mass_spectrometry.zip`：666,754 字节，蛋白丰度参考。

这些包只核实元数据和公开入口，尚未下载逐样本审计。正式使用前要确认 donor、来源、极化条件、时间与表达矩阵的一一对应。若配对元数据不齐，先输出目标不确定性和缺项，不将不同批次简单相减。

## 本方案的推断边界

供者和条件未配对不等于同细胞前后因果轨迹；单基因终点不能识别任意双基因互作或顺序效应。三类升级分别需要新供者、Th2 条件下的扰动、组合或时序干预证据。这些是实验设计层面的限制，不能靠增加参数或交叉模型一致性消除。
