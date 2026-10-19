# 仿真预研（2026-10-14 至 10-19）

SO101-Nexus `MuJoCoPickAndPlace-v1` 仿真中的抓放任务：单个 25mm 红色方块放到蓝色圆盘（半径 2.5cm）上；每个种子只改变布局（方块位置与朝向、圆盘位置），颜色、光照、相机、初始姿态与指令固定。双相机（顶视+腕部）320×240，6 维关节状态，30Hz。这些结果不作为毕设正式基线（正式基线在真机上重建），用于确定方法与评测口径。

## 结论

| 结论 | 依据 |
|---|---|
| 原 SmolVLA（冻结 VLM，只训 Expert）在 98 万帧脚本示教上训练 1 万步，闭环只有约 7%（13/200）；训练帧与验证帧离线误差相同，属欠拟合，且离线误差下降不带来闭环提升 | [so101-scripted-data.md](so101-scripted-data.md) |
| DAgger（在策略访问的状态上由脚本专家接管 50 步作为动作块）再训 5000 步：约 68%（135/200），结构不变 | [so101-dagger.md](so101-dagger.md) 第 3–5 节 |
| 动作块策略的 DAgger 不能沿策略轨迹逐步取专家标签（专家参考点不跟随机器人）；专家接管标签在专家自身轨迹上与原专家逐位一致 | 同上第 3 节 |
| 目标条件化分层结构（目标嵌入+规划头）：真值目标上限 149/200、规划头 127/200，与原结构 135/200 无显著差别；暂停 | [so101-goal-conditioned-design.md](so101-goal-conditioned-design.md)、[so101-dagger.md](so101-dagger.md) 第 6 节 |
| 评测：EGL 渲染在同一状态下有 0–2 像素差 1 灰度级，闭环放大后同一种子结果不可逐次复现（100 种子约 ±2–5）；比较须 ≥100 种子 × 2 次并做逐种子配对检验 | [so101-goal-conditioned-design.md](so101-goal-conditioned-design.md) 第 11.3 节 |
| 早期：作者原始示教回放、仿真微调、单场景过拟合、真机到仿真差异（相机朝向/外观、初始关节） | [so101-sim-policy-diagnosis.md](so101-sim-policy-diagnosis.md)、[so101-real-to-sim-diagnosis.md](so101-real-to-sim-diagnosis.md) |

测试集（种子 200000–200049）从未使用。
