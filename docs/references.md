# 参考资料

本项目用到或值得回头查阅的论文、博客和 issue。每条资料包括：讲了什么、关键数字（摘自原文）、与本项目的关系。链接和编号都核对过（2026-10-24）。详细的数字对比见 [notes/vlm-finetuning-evidence.md](notes/vlm-finetuning-evidence.md)。

## 微调方式：冻结 VLM / LoRA / 全量微调

- **Dream Machines：π0.5 在真实制造任务上的微调实测**（博客，2026-09）
  https://dream-machines.eu/blog/pi05-fine-tuning
  - 双臂 TRLC-DK1，把执行器从纸盒移到托盘孔里；每个设置测 40 次，约 ±15 个百分点的噪声。
  - 全量微调 76%，LoRA rank 256 为 52.5%，rank 64 为 45% / 32.5%。作者推测原因是这种机器人形态不在 π0.5 的预训练数据中。没有测试冻结 VLM。
  - 只用 PaliGemma 初始化（没有机器人预训练）为 0%。
  - 数据：1 个场景 33%，5 个场景 63%；4 h / 10 h / 21 h 分别为 63% / 64% / 76%；再加 1 h 动作平稳的高质量数据，为 90%。
  - 人工干预数据：28% → 88%（240 次运行，其中 212 次有干预）。
  - 部署：同一模型只调 RTC 参数，76% → 93%（新块引导步数 16 → 20）；同步推理只有 53%。
  - 验证集误差无法区分好坏 checkpoint。
  - 与本项目：实验四改为全量微调的主要依据之一；实验三离线误差下降但真机没有提升，与这里的观察一致。

- **Ferchau 等：On the Efficiency of LoRA Fine-Tuning for Vision-Language-Action Models in Industrial Robotic Manipulation**（arXiv 2607.10172，ICANN 2026）
  https://arxiv.org/abs/2607.10172
  - π0，UR5e，4 个装配任务，每个任务 200 段示教，每种设置测 20 次，指标为平均任务进度（ATP）。
  - 全量微调 0.76，LoRA r=32 为 0.74（视觉编码器全量训练），**冻结 VLM、只训练动作模块 0.15**，视觉编码器冻结 0.14。
  - 与本项目：实验二、三用的正是"冻结 VLM、只训练动作模块"这种设置；也是实验四的依据。

- **openpi issue #1060：π0.5 的 LoRA 和全量微调，训练 loss 和离线误差相同，闭环 0/150 对 60/150**
  https://github.com/Physical-Intelligence/openpi/issues/1060
  - 仿真舀粉任务。评论指出两边的学习率和 EMA 设置不同，差距暂时不能全归因于 LoRA，对照实验还没做（截至 2026-10-24）。
  - 与本项目：离线指标不能代替闭环评测。

- **Thinking Machines：LoRA Without Regret**（博客，2025-09，John Schulman 等）
  https://thinkingmachines.ai/blog/lora/
  - 针对语言模型：小到中等规模的数据上，LoRA 加在所有层（尤其是 MLP）上、学习率取全量微调的约 10 倍时，效果可与全量微调持平；数据超出 LoRA 容量时会变差；不适合很大的 batch。不涉及机器人。
  - 与本项目：如果以后受显存限制必须用 LoRA，可参考它的设置建议。

- **How Do VLAs Effectively Inherit from VLMs?**（arXiv 2511.06619）
  https://arxiv.org/abs/2511.06619
  - 比较全量微调、LoRA 和只训练动作头对 VLM 原有识别能力的保留程度（只读过检索摘要，细节未核对）。

- **Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control**（arXiv 2512.11921）
  https://arxiv.org/abs/2512.11921
  - SO-101 按按钮任务，视觉编码器冻结与不冻结分别为 74% 和 76%，8 GB RTX 4060。原文没有给出测试次数，模型描述前后不一致，可信度低，只作参考。

## 模型与推理

- **SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics**（arXiv 2506.01844）
  https://arxiv.org/abs/2506.01844 ，博客 https://huggingface.co/blog/smolvla
  - 4.5 亿参数，用 LeRobot 社区数据预训练，支持异步推理。
  - 与本项目：基础模型（`artifacts/smolvla_base`）。参数构成（本项目统计）：VLM 语言模型 2.52 亿，视觉编码器 8640 万，连接层 1180 万，动作模块 9820 万。LeRobot 默认冻结 VLM。

- **Real-Time Execution of Action Chunking Flow Policies**（RTC，arXiv 2506.07339，NeurIPS 2025，Black、Galliker、Levine）
  https://arxiv.org/abs/2506.07339
  - 把下一块动作的生成看作补全问题：推理延迟期间会执行的动作保持不变，补全其余部分。不需要重新训练。
  - 与本项目：真机推理使用 `--inference.type=rtc`。LeRobot 实现中的两个问题（重置后用旧观测、新动作块开头多丢约 11 步）已修复，见 [notes/inference-sync-vs-rtc.md](notes/inference-sync-vs-rtc.md)。

## 交互式模仿学习（纠错数据）

- **DAgger：A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**（Ross、Gordon、Bagnell，AISTATS 2011，arXiv 1011.0686）
  https://arxiv.org/abs/1011.0686
  - 在策略自己访问到的状态上收集专家标签，然后汇总起来重新训练。

- **HG-DAgger: Interactive Imitation Learning with Human Experts**（Kelly 等，ICRA 2019，arXiv 1810.02890）
  https://arxiv.org/abs/1810.02890
  - 由人决定何时接管；只用人接管期间的动作作为标签，解决 DAgger 中人只给标签、却不实际控制系统所带来的安全和标注质量问题。
  - 与本项目：`scripts/dagger.fish` 的采集方式（人接管纠正后交还模型、只训练人控帧）就是这一思路。

- **RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction**（Hu 等，CMU，arXiv 2509.07953）
  https://arxiv.org/abs/2509.07953 ，项目页 https://rac-scaling-robot.github.io
  - 人接管时先把机器人退回到熟悉的状态（恢复），再完成当前子任务（纠正）；报告称性能随恢复动作的数量线性提升，数据采集时间只需原来的 1/10。
  - 与本项目：LeRobot 的 DAgger 录制方式即按 RaC 设计。我们的规则"接管后至少完成当前这一小步再交还"也来自这里。

## 工具

- **LeRobot**：https://github.com/huggingface/lerobot （本项目使用 0.6.1；录制、rollout、DAgger、数据集工具都来自这里）
