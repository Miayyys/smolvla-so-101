# RECAP v3 原始生成场景拟合测试（2026-11-13）

结论：在这批分层抽样中，训练场景主要抓放行为拟合较好，留出场景的闭环泛化明显不足。低离线动作误差不能替代闭环成功；原生成数据的成功定义也允许物体突出盒沿。

## 样本与方法

从 twin v2/v2b 的原始成功、无故意夹空轨迹中，以固定种子 20261113 抽取训练集 10 段、v3 留出验证集 10 段：五类形状 × 左右侧，每格一段。清单在 `runs/so101_recap_v3/generated_fit_cases.json`，于测试前固定，不按策略结果筛选。此结果适用于这批诊断样本，不能作为全体数据集成功率。

按原数据种子重新构建物体几何、质量、摩擦和颜色，核对原 meta 记录，复现原初始关节、物体与盒子摆放。使用本地 v3 best（19000 步）、object 指令、positive 条件、sigma 2 平滑；每段 60 秒、30 Hz，分别执行 30/50 动作后重规划，共 40 段。全程未连接真机。

逐段回放原专家动作验证仿真可解；专家 20/20 满足原有中心入盒判据。同一批场景在两种执行长度下的初始图像逐像素一致。全部 40 条轨迹的状态/动作/指标形状、有限性及视频帧数检查通过。

另在每段原 top/wrist 视频及专家关节状态上，取 approach/descend/close/lift/carry 阶段的最后一帧预测接下来 50 步，与原专家动作比较；每段 5 个观测，共 100 个，超出记录末端的目标截去。每个观测重置模型并固定采样种子，按训练 action std 归一化后对观测、轨迹取均值。采用相同 sigma 2 平滑，因此不是正式训练验证日志的采样口径，也不把误差直接转为成功率。

## 闭环结果

| 执行动作数 | 数据划分 | 抓起 >3 cm | 中心入盒 | 完整入盒 | 原专家完整入盒 |
|---|---|---|---|---|---|
| 30 | train | 9/10 | 9/10 | 4/10 | 3/10 |
| 30 | validation | 3/10 | 3/10 | 0/10 | 3/10 |
| 50 | train | 8/10 | 8/10 | 5/10 | 3/10 |
| 50 | validation | 3/10 | 3/10 | 0/10 | 3/10 |

中心入盒沿用生成器判据：中心在盒内，局部高度 -5 至 30 mm。完整入盒额外要求所有顶点 XY 坐标都在内壁以内。原专家虽然中心入盒 20/20，但两组严格完整入盒各只有 3/10，提示生成器对“成功”的判定偏松；完全入盒的目标需要在生成与标签阶段明确，不能只靠增加训练步数期待模型自动改正示教。

训练/验证差距在 30/50 两种执行长度下都存在，执行长度并未消除它。这里的同步仿真会在推理时暂停物理，未模拟真机 RTC 的异步推理延迟和前缀引导。

## 专家观测下的动作拟合

| 数据划分 | 观测数 | normalized MAE | 肩转 ° | 肩抬 ° | 肘 ° | 腕俯 ° | 腕转 ° | 夹爪百分点 |
|---|---|---|---|---|---|---|---|---|
| train | 50 | 0.0969 | 3.00 | 2.81 | 4.64 | 3.07 | 1.66 | 0.80 |
| validation | 50 | 0.1336 | 2.89 | 4.65 | 6.40 | 3.63 | 4.31 | 1.36 |

训练集动作误差更低，但部分失败的验证场景也有很低的离线误差，例如 bar `validation_7025` 为 0.084。接近、闭合、抬起时的小误差可以改变接触结果，后续观测也会偏离专家轨迹；当前结果支持“闭环泛化不足”，尚不能单独证明是纯粹记忆或确定需要多少更多数据。

## 逐场景中心入盒

| 数据 | 形状 | 侧 | seed | 30 步中心/完整 | 50 步中心/完整 | 动作 MAE |
|---|---|---|---|---|---|---|
| train | bar | left | 7013 | ✓/✓ | ✗/✗ | 0.074 |
| train | bar | right | 1114 | ✓/✗ | ✓/✗ | 0.105 |
| train | key | left | 3074 | ✓/✗ | ✓/✗ | 0.098 |
| train | key | right | 5009 | ✓/✗ | ✓/✓ | 0.102 |
| train | bent | left | 3050 | ✓/✓ | ✓/✓ | 0.067 |
| train | bent | right | 1038 | ✓/✓ | ✓/✓ | 0.112 |
| train | hammer | left | 1016 | ✓/✗ | ✓/✓ | 0.137 |
| train | hammer | right | 1027 | ✓/✓ | ✓/✓ | 0.106 |
| train | dumbbell | left | 7008 | ✓/✗ | ✗/✗ | 0.081 |
| train | dumbbell | right | 3042 | ✗/✗ | ✓/✗ | 0.087 |
| validation | bar | left | 1148 | ✗/✗ | ✗/✗ | 0.136 |
| validation | bar | right | 7025 | ✗/✗ | ✗/✗ | 0.084 |
| validation | key | left | 3058 | ✗/✗ | ✗/✗ | 0.115 |
| validation | key | right | 3001 | ✓/✗ | ✓/✗ | 0.192 |
| validation | bent | left | 7019 | ✗/✗ | ✗/✗ | 0.145 |
| validation | bent | right | 1083 | ✗/✗ | ✗/✗ | 0.173 |
| validation | hammer | left | 1113 | ✗/✗ | ✗/✗ | 0.098 |
| validation | hammer | right | 1134 | ✗/✗ | ✗/✗ | 0.188 |
| validation | dumbbell | left | 7024 | ✓/✗ | ✓/✗ | 0.091 |
| validation | dumbbell | right | 1018 | ✓/✗ | ✓/✗ | 0.115 |

## 查看场景和录像

- [原始场景总览](../media/recap_v3_generated_fit-initial_scenes.jpg)：每行一种形状，列依次为训练左、训练右、验证左、验证右。
- [30 步最终状态总览](../media/recap_v3_generated_fit-final_scenes.jpg)。
- [训练场景成功示例](../media/train_7013-preview.mp4)。
- [验证场景失败示例](../media/validation_7025-preview.mp4)。
- 详细 JSON/CSV：`runs/twin/recap_v3_generated_fit/`。
- 每段双相机预览、最终图和完整 30 Hz 轨迹：`runs/twin/policy_recap_v3_generated_{30,50}/`。预览仅在重新规划时采帧。

## 复现

```bash
env PYTHONPATH="$PWD" MUJOCO_GL=egl .venv-gs/bin/python scripts/twin_policy_eval.py \
  --model runs/so101_recap_v3/best --out runs/twin/generated_fit_repeat_30 \
  --cases-json runs/so101_recap_v3/generated_fit_cases.json \
  --teacher-forced-stats runs/so101_recap_v3/stats.json --execute 30
```

50 步复测将 execute 改为 50，并使用新输出目录。动作拟合部分不依赖执行长度，无需重复执行。
