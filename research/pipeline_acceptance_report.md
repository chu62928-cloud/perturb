# Pipeline 验收记录（2026-09-01，修订版）

状态：**通过（D1 数据契约、连续状态接口与闸门骨架）**。

- 编排环境：`e3_pipeline`，Python 3.11.16，Snakemake 8.30.0，NumPy 2.4.6，SciPy 1.17.1，AnnData 0.12.19，Scanpy 1.11.5。
- 依赖：`pip check` 无未解决依赖。
- 工作流：Snakemake `preflight → audit → prepare_pilot → freeze_data` 实际运行完成，且可强制重跑。
- 数据注册：7/12 个当前存在文件登记为完整；D1 为开发供者，D2 为暂定外部供者，D3/D4 未完整文件不进入训练或正式测试。
- D1 试点：4,661 个候选，生成 96 个结构试点（64 个 core、32 个 challenge）。其中只有 473 个候选有抽样 NTC 表达分箱；guide 校订、效应强弱、功能类别和刺激依赖仍待补充。
- 试点隔离：已保留历史 `pilot_96.json`，并生成 `pilot_64_primary_v1.json`（64 个、响应盲）和 `pilot_32_challenge_v1.json`（32 个、仅压力测试），两者无重叠；主要统计只允许使用 primary_64。
- 数据契约：`data_contract_frozen_v1.json` 已生成，当前为 `DATA_CONTRACT_READY`（未宣称 `DATA_VALID`）；记录注册表、两层试点哈希，并明确 `d2_responses_used=false`。待 guide 映射率、基因顺序哈希、归一化哈希和划分哈希证据补齐后才可升级。
- 公共摘要：已从固定 GitHub 提交 `aa5c84a973c0e1a090b0072dc5b080bf7fbbed38` 下载并审计 guide 库 26,504 行、guide 效率 73,765 行、DE 摘要 33,983 行和外部 Th1/Th2 参考 714,594 行。DE/guide 效率摘要目前只作审计，不用于 D2 调参或锁定测试样本选择。
- 工程改造：连续潜空间锚点/200近邻区域/1.25倍缓冲留出、D1 分层 Bootstrap、`MODEL_GLOBAL_VALID`/`MODEL_STATE_VALID` 双闸门、几何匹配封存后再读取第二步结果、多目标安全契约均已实现。
- 非深度流水线：基线、分布指标、支持度、三类留出、组合性闸门、模型分歧、锁定发布、束搜索及新增阶段接口 smoke 通过；远端 `pytest` 13 项通过，`pip check` 无未解决依赖，Snakemake 已完成且当前无待执行规则。

注意：`prepare-pilot` 遇到大型 backed CSR 的高级索引兼容问题后，已改为 AnnData 0.12.19 + slice-first 路径。该改动仅影响读取方式，不改变原始 `.h5ad` 文件。

## 尚未放行的阶段

`MODEL_GLOBAL_VALID`、`MODEL_STATE_VALID`、`COMPOSITION_VALID`、`PLANNER_VALID` 和 `EXPERIMENT_VALID` 尚未声明。原因是单步效应矩阵、冻结程序基因集、连续状态实际嵌入和独立供者验证尚未完成；当前 96 基因结果中 challenge_32 仍只能作为压力测试，不能解释为已完成 Th2→Th17 序贯预测。

## 下一步

1. 在不读取 D2 的前提下完成 guide 校订、D1 单步效应矩阵、功能/刺激依赖注释和 8 套程序基因集冻结。
2. 基于 D1 NTC 生成 50 维连续状态区域和三类严格留出，先跑无变化、条件均值、线性残差、`pert2state` 和 STATE 零样本。
3. 依次评价全局与状态依赖闸门；只有两个闸门同时通过，才启用第二执行器和代理组合性。
4. 组合性匹配清单先封存，再读取第二步真实响应；至少 50 个有效三元组、20 个第二动作基因和 3 个状态区域后才评估 `COMPOSITION_VALID`。
5. D2 只做一次性暂定锁定测试；D3/D4 完整后进行最终外部验证。数据盘不足时不恢复大文件下载。
