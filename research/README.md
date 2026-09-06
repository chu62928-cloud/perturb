# 实施方案的核查材料

主文档是上一级的 `perturb_implementation_plan.md`。`perturb_implementation_plan.html` 是从同一 Markdown 生成的离线预览，数学字体已经内嵌；在不支持数学公式的 Markdown 阅读器中，可用浏览器打开预览版。

## 阅读顺序

1. 主文档第1节：结论与分阶段路线。
2. 第5节与第11节：可立即实施的任务、交付和门槛。
3. 第6–9节：首版完成后的验证、组合与可选时序扩展。
4. 本目录的 `data_and_target_evidence.md`、`model_evidence.md`、`novelty_evidence.md`：调研依据和未解决问题。

## 事实与验证范围

`server_audit.json` 是本次只读核查记录。服务器上的三个 D1 文件可打开且抽样可读，D2_Stim8hr 文件截断。全文件源端摘要尚未核对，不能把抽样可读写成逐字节完整性证明。本次没有修改、删除或续传服务器数据。

本次未运行候选排名、训练模型或生物实验。实施规模与门槛是方案建议，不是实测性能。独立审查记录保存在 `outputs/researchwrite/th2_th17/qa_logs/`；审查分数衡量方案质量，不是成功概率。

## 重新生成公式预览

在项目根目录运行：

```powershell
npm ci --prefix research/preview_tools
node research/render_plan.cjs
```

依赖固定于 `preview_tools/package-lock.json`，浏览器使用本机 Microsoft Edge，脚本默认路径为 `C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe`。换机器时需按实际路径调整。

脚本以严格模式解析公式，生成主目录的 HTML、`qa_preview/` 中的截图及 `validation.json`。报告记录源文件 SHA-256、公式数、连续编号、字体加载、浏览器错误及溢出检查。修改 Markdown 后需重新生成 HTML，并重新检查公式截图。

浏览器通过与人工看图只能核验排版；公式的科学含义仍需结合正文与研究审查。报告不包含服务器密码或访问密钥。
