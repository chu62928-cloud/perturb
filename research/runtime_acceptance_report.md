# CD4 T细胞序贯扰动：运行环境验收记录

核查日期：2026-09-01；服务器显卡：NVIDIA GeForce RTX 4080 SUPER，显存 32760 MiB，驱动 580.142。

## 结果摘要

| 执行器 | 环境与关键版本 | 权重/检查点 | 最小计算结果 |
|---|---|---|---|
| STATE | `e3_state`，Python 3.12.4，PyTorch 2.7.0+cu128，STATE `9bbfe78a434a55205e4de834e1ea99f85f7a3add` | 官方 `ST-HVG-Replogle` `last.ckpt` | GPU加载；输入 `[1,64,2000]`，输出 `[64,2000]`；峰值显存约214 MB |
| STACK | `e3_stack`，Python 3.10.20，PyTorch 2.11.0+cu128，STACK `cacc2e4b09435c3e536d46237d10b50f222dd144` | 官方 `Stack-Large-Aligned/bc_large_aligned.ckpt` | GPU加载；输入 `[1,8,15012]`，输出 `[1,8,15012]`；嵌入 `[1,8,1600]`；峰值显存约3.53 GB |
| PRiMeFlow | `e3_primeflow`，Python 3.12.4，PyTorch 2.7.0+cu128，PRiMeFlow `443295258e364e394b60de27a861d8d1cf3b1fa6`，JAX 0.6.1 | 自建缩小 `DynamicsMLP` 检查点（官方未公开预训练权重） | 输入/输出 `[4,8]`；新实例回载最大绝对误差0 |
| GEARS | `e3_gears`，Python 3.10.20，PyTorch 2.7.0+cu128，GEARS `f374e43e197b295016d80395d7a54ddb81cc6769`，`cell-gears==0.1.2`，`torch-geometric==2.6.1` | 自建小型检查点 | 单基因扰动索引0；输入 `[8,1]`，输出 `[1,8]`；回载最大绝对误差0 |
| pert2state | `e3-th-actuator`，`pert2state-model==0.0.1` | 不需要深度权重 | 6×3矩阵拟合与预测，输出均为有限值 |

五个环境的 `pip check` 均为 `No broken requirements found`。完整依赖锁定见同目录下各环境的 `requirements.lock.txt`、Conda显式规格及环境导出文件；本轮没有把模型依赖混入原有 `e3-th-actuator`。

## 路径与限制

- 远端运行根目录：`/root/autodl-tmp/CRISPR_perturb_runtime/`。
- 官方权重位于 `models/huggingface/`，许可文件随 STACK 权重保存；不保存任何令牌。
- 7/12个原始数据文件审计报告：`research/server_audit_current.json`；5个缺失文件合计约641.13 GiB。
- 当前数据盘约260 GiB可用，低于继续完整下载所需的约800 GiB门槛；不删除已有数据。
- GitHub与Hugging Face下载只在临时子进程中启用 `/etc/network_turbo`，未写入持久配置。

本记录只证明环境、权重/检查点和最小计算可运行，不证明模型已经适用于Th2→Th17序贯扰动；正式适用性必须由D1开发、D2外部供者的严格留出基准和实验验证决定。
