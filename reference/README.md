# reference：原样归档的代码（不维护，不保证可直接运行）

来源：QVLA 仓库，分支 `experiment/so101-goal-conditioned` @ `949f701`（其中 RK3588 部分与 `experiment/rk3588-runtime-optimization` @ `ef21894` 相同）。导入路径仍是 `qvla.*`，需要时按 [../README.md](../README.md) 的对应表改写后再用。

| 目录 | 内容 | 对毕设的用途 |
|---|---|---|
| `qvla_rk3588/conversion/` | SmolVLA 视觉编码器、prefix、Expert 的 ONNX/RKNN/RKLLM 导出与编译、各种探查脚本（原针对 `lerobot/smolvla_libero`） | RK3588 部署链路的起点 |
| `qvla_rk3588/runtime/`、`hardware/` | 板端运行时、算子/精度探查 | 板端推理与测速 |
| `qvla_rk3588/quantization/`、`haq/` | 实 INT8 线性层、混合精度、HAQ 搜索框架 | 量化与混合精度搜索 |
| `qvla_rk3588/docs/` | QVLA 的 `project-route.md`（完整实验路线）、`deployment.md`、`results.md` | 前期 RK3588/量化结论与踩坑记录 |
| `qvla_so101_early/` | 早期 SO101 探索：作者数据回放、仿真微调、单场景过拟合、真机到仿真诊断、SO100 真机微调流水线；`haq_offline_eval.py` 全文；对应配置 | 真机数据处理与诊断思路 |
| `qvla_so101_pipelines/` | 预研时无人值守运行的流水线脚本（路径指向 QVLA） | 复现记录 |
