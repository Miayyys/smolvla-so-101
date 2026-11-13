# 已归档的轻量结果

- `v5-key-outcomes.csv`、`v5-doll-outcomes.csv`：2026-11-04 真机录像逐段核验的成败记录；未知结果不计入分母，钥匙 13/20、玩偶 17/21。不是刚完成的新一轮测试。
- `recap-v2-v3-twin-comparison.json`：两版模型在固定代理场景中的本地仿真对比，包含 30/50 步两种执行长度。
- `recap-v3-generated-fit.json`、`.csv`：原生成数据场景的训练/留出集拟合与闭环结果。
- `recap-v3-generated-cases.json`：测试前固定的分层抽样清单，含来源数据集、原 episode 和合并行号。

本目录只含小体积结果，不含模型、完整训练数据、相机缓存或全量轨迹。JSON 中的运行路径是原实验位置；代表性视频/场景图已单独复制到 `docs/media`，详见实验报告。数据和评测口径见 [当前状态](../project-status.md)。
