# 本机训练速度与显存（RTX 4060 Laptop 8 GB）

实验二训练期间做的速度和显存检查，用来决定以后的训练设置。

## GPU 是否跑满（2026-10-22 19:53，实验二第 7960 步左右）

采样 12 s：
- GPU 使用率 97–100%，显存 7.7 GB。
- 功耗约 83 W，功耗上限 140 W；SM 时钟 2520 MHz；温度 72 °C。
- 没有出现功耗限制或温度降频。
- 笔记本平台功耗模式为 `performance`；训练进程 CPU 占用约 250%（整机约 16%）。

图片解码在后台线程，不拖慢 GPU；每 500 步一次验证只要几秒。所以训练速度受 GPU 算力限制，没有硬件上的余量可挖。

## torch.compile 测速（2026-10-22 21:30）

用户问 LeRobot SmolVLA 配置里的 `compile_model` / `compile_mode="max-autotune"` 有没有用。
- 实验一、二都没有开启：`compile_model` 为 false，`so101vla/train.py` 也不调用 `torch.compile`。
- LeRobot 开启这个选项时，会编译 `forward`（训练）和 `sample_actions`（推理），见 `modeling_smolvla.py:534-537`。

**方法**：
- 实验二在第 10000 步存档后停训，用 `scripts/bench_compile.py` 测，环境为 torch 2.7.1+cu118、triton 3.3.1。
- 只编译训练用的 `policy.model.forward`。所有模式用同一份 `last/` 权重和同一组 40 个训练 batch（每个 16 帧）。
- 第一个 batch 用固定的噪声和时间步，算 loss 和梯度范数。
- 然后预热 8 个 micro-batch，再计时 24 个，每 4 个做一次 AdamW 更新（和训练相同）。

| 模式 | 每个 micro-batch | ×4 = 每步 | 显存峰值 | 首次调用（含编译） | 首 batch loss / 梯度范数 |
|---|---|---|---|---|---|
| 不编译（当前） | 0.777 s | 3.11 s | 7.06 GB | 1.5 s | 0.03214 / 0.7167 |
| `default` | 0.775 s | 3.10 s | 6.34 GB | 63.9 s | 0.03173 / 0.7180 |
| `max-autotune-no-cudagraphs` | 0.971 s | 3.88 s | 6.34 GB | 158.4 s | 0.03171 / 0.7197 |
| `max-autotune`（LeRobot 默认） | 失败：`CUDA error: operation not permitted when stream is capturing` | | | | |

**结论**：在本机编译不提速，所以不开启。
- `default` 和不编译一样快；`max-autotune-no-cudagraphs` 慢 25%；带 CUDA graph 的 `max-autotune` 直接报错。
- bf16 下，编译后的 loss 相对差约 1%，梯度范数相差 0.2–0.4%。
- 测速时每步 3.11 s，比实际训练的 2.86 s 慢，因为测速脚本没有把 JPEG 解码放到后台线程。所以这些数字只用来比较不同模式。
- 为什么不提速没有逐层分析。推测是 GPU 已经满载，时间主要花在矩阵乘和注意力上，编译能合并的逐元素运算占比小（未验证）。
- 推理时编译 `sample_actions` 的效果未测量。

## 显存不足

测速后第一次续训时报 `CUDA out of memory`。当时 VS Code（234 MiB）和 Chrome（83 MiB）也占着显存；Chrome 是在上一轮训练开始后才打开的。训练本身约需 7.07 GB，而显卡可用 7.7 GB。

**处理**：启动命令加环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。它只改变 PyTorch 分配显存的方式、减少碎片，不改变计算。加上后训练正常，显存峰值从 7.07 GB 降到 6.90 GB，速度仍为 2.86 s/步。
