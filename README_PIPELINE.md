# CD4 Perturb-seq 流水线骨架

这是 V0/V1 的可审计执行骨架。它把原始 `.h5ad` 文件视为只读输入，并将数据注册、试点基因、简单基线、模型评价和两步规划拆成可恢复的步骤。深度模型仍由独立环境提供；本目录不把 STATE、STACK、PRiMeFlow 或 GEARS 的依赖混入编排环境。

## 当前可运行入口

```bash
export PYTHONPATH="$PWD/src"
python -m cd4perturb.cli preflight --config config/config.json
python -m cd4perturb.cli audit --config config/config.json \
  --audit /root/autodl-tmp/CRISPR_perturb_runtime/data_audit/current.json
python -m cd4perturb.cli audit-csr --config config/config.json \
  --paths config/d2_paths.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/development_D2_v1 \
  --output-dir /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/development_D2_v1/metadata/d2_csr_audit_v1 \
  --block-rows 1024
python -m cd4perturb.cli activate-roles --config config/config.json \
  --audit-summary /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/development_D2_v1/metadata/d2_csr_audit_v1/d2_csr_audit_full_summary.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/development_D2_v1
python -m cd4perturb.cli prepare-pilot --config config/config.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
python -m cd4perturb.cli freeze-data --config config/config.json \
  --primary /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/pilot_64_primary_v1.json \
  --challenge /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/pilot_32_challenge_v1.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
python -m cd4perturb.cli audit-public --config config/config.json \
  --public-root /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/public/<commit> \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
python -m cd4perturb.cli guide-qc --config config/config.json \
  --paths /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/d1_paths.json \
  --guide-library /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/public/<commit>/sgrna_library_metadata.suppl_table.csv \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
python -m cd4perturb.cli gene-order --config config/config.json \
  --paths /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/d1_paths.json \
  --pilot /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/pilot_96.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
python -m cd4perturb.cli ntc-latent --config config/config.json \
  --paths /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/d1_paths.json \
  --genes /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/shared_gene_order_v2.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
python -m cd4perturb.cli effect-matrix --config config/config.json \
  --input /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/d1_paths.json \
  --genes /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/shared_gene_order_v2.json \
  --guide-qc /root/autodl-tmp/CRISPR_perturb_runtime/pipeline/metadata/guide_qc_v1.json \
  --output /root/autodl-tmp/CRISPR_perturb_runtime/pipeline
snakemake --snakefile workflow/Snakefile --cores 1
```

`audit` 会生成 `metadata/dataset_registry.json`；D2 只有在 `audit-csr` 覆盖完整 `X/indptr`、`X/indices`、`X/data` 并通过后，才由 `activate-roles` 标为开发。新协议把 D1 标为一次性二级确认、D3/D4 标为最终外部测试。原始文件大小或 HDF5 可读性不满足时，记录为阻塞状态，不进入响应分析。

## 研究闸门

研究状态严格按 `DATA_VALID → MODEL_GLOBAL_VALID → MODEL_STATE_VALID → COMPOSITION_VALID → PLANNER_VALID → EXPERIMENT_VALID` 递进。全局和状态闸门都只使用 `primary_64`；`challenge_32` 是 D2 开发响应驱动的压力测试，只单独报告，不能进入主要统计量。D2 自助法为 `gene → guide → cell` 或 `gene → region → guide → cell`，只表示开发供者内技术/状态层级不确定性，不能解释为供者泛化。

数据阶段命令通过不可变角色清单记录 `development_donor`、`response_donors_used` 和角色哈希；`freeze-data` 只有在指南校订、归一化、基因顺序、D2 效应、完整 CSR 审计和三类划分证据齐全时才会输出 `DATA_VALID`。

`prepare-pilot` 已实现 D2 开发试点：保留历史产物，并生成响应盲 `primary_64`、预先冻结的 256 基因挑战池、`challenge_32` 和对应 manifest。前者只依照扰动前表达与覆盖度选择；挑战集只用于压力测试，不能进入主要闸门统计。

`ntc-latent` 从 D2 NTC 的固定 2000 特征按 CP10K 对数变换拟合 50 维 PCA；`state-regions` 使用有界近邻查询（不构造全距离矩阵），每个条件选择 10 个高密度、相互分离的锚点，以第 200 近邻距离定义区域，并在 1.25 倍半径处设置训练缓冲区。Leiden 仅用于描述和图表，不作为规划器状态单位。

`fit-baselines` 接受统一输入中的 `x_control`、`y_perturbed` 和可选 `gene_effect`、`pert2state_prediction`，登记无变化、条件均值、基因效应转移、岭回归和 `pert2state`。`state-adaptation` 只接受 D2 `train/validation` 的 `primary_64` 指标：若 STATE 零样本在标准化 RMSE 与方向准确率上都劣于最佳简单基线，记录“当前无适配依据”；否则按预注册的 output layer、last layers、LoRA 三项比较，拒绝包含测试指标的输入。

目标契约为多目标安全约束：`Th17` 上升、`Th2` 下降，且应激、凋亡和泛活化不超过 D2 NTC 第 95 百分位；综合分数为 `1.0*Th17 - 1.0*Th2 - 0.5*stress - 0.75*apoptosis - 0.5*general_activation`。组合性必须先完成只依赖几何的匹配清单并封存，再读取第二步真实响应；匹配清单一旦改变，后续评分应拒绝。

`audit-public` 只核查固定 GitHub 提交的公共元数据。汇总效应和 guide 效率不自动进入 D2 选择或调参；D2 guide 校订只能使用固定提交库，D1 只保留为一次性二级确认。

当 D2 的固定基因顺序与状态空间准备好后，使用 `effect-matrix`（`--input` 为 D2 路径 JSON，`--genes` 为基因顺序 JSON）生成逐 guide 与稳健基因级效应；新命令通过角色清单拒绝错误供者路径，并以文件为单位、只读取选定列。使用 `state-regions`（紧凑 NPZ，包含 `latent`、`conditions`）生成连续状态区域；如果同一输入还包含组合性数据，状态条件可放在 `latent_conditions`。基线、评价、组合性和规划阶段均要求显式输入，缺失输入时直接失败，不会伪造模型通过状态。
