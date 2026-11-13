# RECAP：按优势加好坏标签训练（2026-11-11 起）

依据 π\*0.6 的 RECAP（arXiv 2511.14759，第 IV-B、V-B 节）。截至 2026-11-13，已完成 Monte Carlo 价值函数训练、优势标签生成，以及 v2/v3 策略微调和闭环测试。当前真机表现最好的仍是 V5，RECAP 没有验证出优于基线的效果。本文按日期保留过程记录，当前状态见 [项目状态](../project-status.md)。

这里实现的是基于回报价值估计和优势条件化模仿学习的离线策略改进，不包含 PPO/SAC 或在线策略梯度更新。negative 样本仍参与条件模仿损失，不能理解成直接惩罚失败动作；30% 条件 dropout 也会让失败样本参与无条件训练。故意夹空数据的影响尚未通过消融实验确认。

## 标签放在哪里（`so101vla/advantage.py`）

按论文的做法：标签是文字（"Advantage: positive" / "Advantage: negative"），嵌入方式和任务指令相同，放在指令之后、状态之前，单独成一个注意力块：

```
[图像 + 指令]（块 0，双向） | [标签]（块 1） | [状态]（块 2） | [动作]（动作专家）
```

- 图像和指令看不到标签，所以它们的特征和标签无关；状态和动作能看到标签。这对应论文的 "only the action log-likelihoods are affected"。
- 标签被随机去掉时，这几个 token 仍在序列里，但全部标成填充：不占位置编码，也没有 token 看它们，结果和原版 SmolVLA 相同。
- 推理时默认写 positive，对应论文的 β = 1，不做无分类器引导。
- 不新增参数。checkpoint 目录里有 `advantage.json` 就表示带标签训练过：
  - `load_policy` 加载时会自动打开标签；
  - `so101vla.rollout`（真机测试）也会自动打开。
- 实现方式是给 `VLAFlowMatching.embed_prefix` 打补丁，只对设置过 `advantage_tokens` 的模型生效。

## 训练（`so101vla/train.py`）

- 配置项 `"advantage": {"labels": "<npz>", "dropout": 0.3}`。npz 里的 `indicator` 是每个数据集行一个 0/1 值。
- 有这个配置时，纠错段里模型控制的帧也参与训练，因为标签已经说明了要不要模仿它。每个动作块都取到这段（裁剪后的）结尾为止，不再在接管处截断。
- 每帧以 30% 的概率去掉标签。去不去由取样本的同一个随机数生成器决定，所以从中断处恢复训练时结果不变。
- 验证集都是示教，按默认的 positive 评估。

## 验证（2026-11-11，v5 权重，本机 8 GB GPU）

- 同一组噪声下：
  - 去掉标签的结果和原版 SmolVLA 相比，float32 下最大差 5.6e-6，bf16 下最大差 0.011，都是不同序列长度带来的数值误差；
  - positive 和 negative 的动作最大相差约 0.93；
  - 一个批次里混着三种标签，每行结果都和单独算时完全相同。
- 图像和指令里的有效 token：positive、negative 两种情况下特征逐位相同，float32 和 bf16 都是。
  - 坑：指令的填充 token 不能算进比较。它们不看任何 token，softmax 退化成对全部 key 的平均，于是会混进标签。但没有 token 会看它们，所以不影响结果。
- 训练冒烟测试：12 段数据，随机标签，只训动作专家，20 步。
  - 损失和验证误差都正常，`best/` 里写出了 `advantage.json`；
  - 重新加载后标签自动打开；
  - 真机测试的加载钩子对 v5 不生效，对新 checkpoint 生效。

## 数据：`so101_recap_v1`（`scripts/recap_dataset.py`）

共 540 段、358488 帧，只收成败已知的数据：

| 类别 | 任务 | 段数 | 成功 |
|---|---|---|---|
| 示教 | 钥匙 / 玩偶 | 98 / 100 | 全部 |
| 纠错（模型 + 人接管） | 玩偶 / 钥匙 | 30 / 29 | 28 / 29 |
| 真机测试录像（只有模型） | 钥匙 / 玩偶 | 80 / 103 | 26 / 55 |
| 孪生脚本专家 | object | 100 | 全部 |

- 成败的来源：
  - 纠错段：`runs/dagger/*.csv`；
  - 测试录像：`runs/real_eval/<名字>.csv`；
  - `rollout_full_v5_doll_raised` 第 15、16 段没有标注，成败记为 null，归入 test 划分，不参与训练。
- 每段的来源和成败写在 `meta/recap_outcomes.json`。
- 合并要求各数据集特征相同，所以每个来源都加了 `intervention` 列：
  - 测试录像全为 False；
  - 孪生全为 True；
  - 带标签的副本放在 `artifacts/datasets/recap_tagged/`。
- `so101_pilot_v3` 有 episode 元数据指向不存在文件的老问题（见 `merge_corrections.py`）。
  - 处理办法：数据和视频用硬链接，`meta` 复制一份后修正，原数据不动。
- 核对：每个来源各抽 1–6 段，动作逐值相同、帧数相同、`intervention` 符合预期；行号连续。视频对齐在 prepare 时逐帧检查。
- 两个划分：
  - `configs/so101_recap_v1_partition_policy.json`：验证集是实验五的 11 段示教；
  - `configs/so101_recap_v1_partition_value.json`：在此基础上，再按种子留出 15% 的测试录像和孪生段（42 段），用来检查价值函数在没见过的段上的表现。

## 价值函数（`so101vla/value.py`，配置 `configs/so101_value_v1.json`）

- 回报：
  - 每步 -1；成功结束时为 0，失败结束时为 -C（C = H = 60 s × 30 帧）；不打折扣。
  - 除以 (H + C) 归一化：成功段的值落在 [-0.5, 0]，失败段落在 [-1, -0.5]。
  - 最长的一段是 59.8 s，在 60 s 以内。
  - 用每段的结束时刻代替"成功时刻"。v5 的测试录像大多是物体进盒后人按键结束的，但 pilot v2/v3 的测试是跑满时限才停，这部分有误差。
- 模型：
  - 用 SmolVLA 的视觉语言模型，输入图像、指令和状态，即策略的前缀，权重从 v5 初始化；
  - 取前缀最后一个 token（状态 token，它看得到其他所有 token）的最终隐藏状态，经 LayerNorm 和线性层得到 201 个区间上的 logits，用交叉熵训练。
  - 推理时要用 `use_cache=True`，这样每一层都对前缀做自注意力；否则交叉注意力层会去找动作专家的 token。
  - 训练整个视觉语言模型，包括视觉编码器；新加的输出头学习率是 10 倍。
- 打标签：
  - 50 步优势：A = (G_t − G_{t+50}) + V(t+50) − V(t)；离段尾不到 50 步时用 G_t − V(t)。
  - 每个任务取训练帧优势的 60% 分位数作阈值，即约 40% 的帧为 positive。
  - 纠错段里人控的帧一律记为 positive。
  - 输出 `labels.npz`，每个数据集行一个标签，另存价值、优势和回报。
- 风险：价值函数如果把训练段记得太牢，V(t) 就会贴近该段的真实回报，训练段的优势都趋近 0，标签只剩噪声。
  - 论文用大模型加网络图文数据联合训练来防止过拟合，我们数据少。
  - 现在用留出段做早停，打完标签后对比训练段和留出段的优势分布。如果差别很大，就改成交叉拟合：K 个价值函数各自给自己没见过的那一折打标签。
- 本地冒烟测试（12 段，伪造成败，只训语言部分）：
  - 训练、验证、打标签都跑通；
  - 显存峰值 5.3 GB（batch 2）；
  - positive 约 40%，没有 NaN。

## 缓存缩小（`cache_resize`）

- 帧在存进缓存前，先按 SmolVLA 自己的 `resize_with_pad` 缩放并补边到 512×512。
- 训练时再经过 `resize_with_pad`，因为尺寸已经一致会直接原样返回。和原来比，只多了 uint8 取整和小图 JPEG 的误差。
- 补边的行数（顶视 224 行、腕部 128 行）都是 8 的倍数，JPEG 的 8×8 块正好对齐边界，不会把黑边糊进画面。
- 大小：
  - 顶视每帧从 261 KB 降到 64 KB；
  - 腕部从 80 KB 降到 60 KB。
- 全部数据约 23 GB + 21 GB：顶视放在 `/dev/shm`，腕部放在网盘 `/root/sj-fs`。网盘随机读约 1450 次/s，训练需要的远低于这个数。

## 上传：先缩小视频（`scripts/shrink_videos.py`）

- 本机上行只有约 0.6 MB/s，开多路并行也不会更快。`so101_recap_v1` 的视频有 11.3 GB，原样上传约要 5 h，这段时间 GPU 照样计费。
- 做法：
  - 用模型自己的 `resize_with_pad` 先缩放好，再切掉补边：顶视 1280×720 → 512×288，腕部 640×480 → 512×384。
  - 用 SVT-AV1 重新编码：crf 24，每 240 帧一个关键帧；LeRobot 录制时是每 2 帧一个。每帧的 pts 和时间基都保留，所以 episode 元数据不用改。
- 已缩放的帧再经过一次 `resize_frame` 结果不变（已验证）。所以用缩小后的数据训练，和用原数据训练相比，只多了这一次重编码的损失。
- 抽测一个文件：
  - 每帧大小：顶视 1.5 KB，腕部 2.2 KB；原来平均约 20 KB 和 12 KB。
  - 平均绝对误差约 2.4–2.7（8 位灰度）。crf 30 时是 2.6–2.8，差别不大，可见误差主要来自 4:2:0 色度采样，而不是压缩档位。
- 每个文件都会核对：所有帧的帧数和 pts 一致，每 20 帧抽一帧算平均误差，要求 < 4。结果写在 `meta/shrink_report.json`。
  - 第一次运行时每个进程把整个文件的帧都放在内存里，8 个进程把本机 15 GB 内存占满，进程被系统杀掉。现在改成边解码边编码，4 个进程；重跑时没有记录的文件会重做。
- 结果（2026-11-11）：
  - 80 个文件，12.0 GB 压到 1.39 GB；
  - 平均误差 2.31，最差的文件 2.65，最差的一帧 3.5。
  - 用 LeRobot 读压缩后的数据，在开头、中间、末尾三处和原数据缩放后的画面对比，误差为 1.7–2.5，说明帧对得上。
- 上传：在服务器上放在网盘 `/root/sj-fs/datasets/so101_recap_v1_512`，换实例后还在，项目里用软链接指过去。一边压缩一边上传。
- 配置改用 `artifacts/datasets/so101_recap_v1_512`。
- smolvla_base 不上传，在服务器上从 hf-mirror 下载，约 1.2 MB/s：
  - 版本固定为 `5e8d12a6`，和本机缓存的元数据一致；
  - `model.safetensors` 的 sha256（7cd549…）和本机相同。

## 服务器流程（2026-11-11 新开的实例）

环境：
- 系统 Python 没有 pip，先用 `/root/miniconda3/bin/python -m pip` 装 uv。
- 其余和 10-25 那次一样：用本机 `uv export` 导出带哈希的依赖清单，Python 从 npmmirror 下载，PyPI 走腾讯镜像，torch cu118 走阿里云。
- 脚本 `/root/sj-tmp/server_env.sh`，装好后 torch 2.7.1+cu118 能用 4090 D。
- v5 权重从网盘 `/root/sj-fs/so101_full_v5/best` 复制过来，sha256 和本机一致。
- SSH 只允许密钥登录。

```bash
cd /root/sj-tmp/so101-vla-edge
.venv/bin/python -m so101vla.value --config configs/so101_value_v1.json --stage prepare
.venv/bin/python -m so101vla.value --config configs/so101_value_v1.json --stage smoke
nohup .venv/bin/python -m so101vla.value --config configs/so101_value_v1.json --stage train > runs/so101_value_v1/train.log 2>&1 &
.venv/bin/python -m so101vla.value --config configs/so101_value_v1.json --stage label        # -> runs/so101_value_v1/labels.npz
.venv/bin/python -m so101vla.train --config configs/so101_recap_v1.json --stage prepare      # 缓存和价值函数共用，字节相同
.venv/bin/python -m so101vla.train --config configs/so101_recap_v1.json --stage smoke
nohup .venv/bin/python -m so101vla.train --config configs/so101_recap_v1.json --stage train > runs/so101_recap_v1/train.log 2>&1 &
```

- 服务器上的情况（2026-11-11）：
  - prepare：
    - 训练帧 320686，验证帧 530，时间戳最大误差 1.8e-6 s，官方解码器比对通过，用时约 20 分钟；
    - 顶视缓存 15.6 GB（`/dev/shm`），腕部缓存 19.4 GB（网盘）。
    - 坑：解码进程里 torch 默认会开满线程，16 个进程 × 16 个线程，4 分钟一段都没做完。改成每个进程 torch 单线程后，约 600 帧/s。
  - 价值函数冒烟测试：batch 16 时显存峰值 20.5 GB，约 1.1 s/步。
  - 20:58 开始正式训练价值函数（`runs/so101_value_v1/train.log`）。第 0 步验证交叉熵 5.37，价值平均绝对误差 0.35。
  - 策略的 prepare 需要先有 `labels.npz`，所以等价值函数训完、打完标签再做。缓存的字节和价值函数那份相同，会写到同一个位置。
  - 自动关机：还没开，要先从平台复制当次的关机 token。
- 策略训练：从 v5 开始，按实验五的方案训练 5000 步，每帧有 30% 的概率去掉标签。
  - 纠错帧不再重复 4 次，因为它们已经强制记为 positive。
  - 模型自己控制的帧带着标签一起训练。

## 价值函数 v1 的结果和标签检查（2026-11-11）

- 训练：
  - 留出段交叉熵最低的是第 1500 步（3.394），第 3000 步提前停止；
  - 第 1500 步的验证：价值平均绝对误差 0.104，按价值判断成败的准确率 85%。
  - 打标签用时约 1.6 h（每秒约 62 帧，GPU 已满载）。
- 阈值（60% 分位数）：玩偶 0.0030，钥匙 0.0045，object 0.0016。
- 检查结果（`scripts/recap_label_report.py`，输出在 `runs/so101_value_v1/label_report.json`）：

| | 训练段 | 留出段 |
|---|---|---|
| 价值平均绝对误差 | 0.060 | 0.136 |
| 优势的标准差 | 0.060 | 0.091 |
| 第一帧价值判断成败的 AUC（测试录像和纠错段） | 0.92 | 0.61 |
| 每段 positive 占比判断成败的 AUC | 0.96 | 0.91 |
| 成功测试录像的 positive 占比 | 0.50 | 0.51 |
| 失败测试录像的 positive 占比 | 0.34 | 0.37 |

- 结论：
  - **有过拟合**：
    - 训练段的误差只有留出段的一半，优势的分布被压窄到约 2/3；
    - 第一帧本来很难看出这段最后会不会成功，训练段上 AUC 却达到 0.92，说明它记住了训练段的结局。
  - **但标签有信息**：
    - 在没见过的段上，每段 positive 的占比仍能把成败分开（AUC 0.91）；
    - 失败段的 positive 占比明显更低，训练段和留出段的数值也接近（0.34/0.50 对 0.37/0.51）。
  - 示教和孪生全是成功段，段内优势很小（标准差分别为 0.02 和 0.009），各自约 40% 的帧为 positive，这部分标签接近随机。
    - 孪生单独是一个任务，阈值只在孪生数据里取，所以 positive 正好是 40%。
    - v2 的故意夹空段应该会拉开孪生段内的差别。
- 可以改进的地方：
  - 交叉拟合：比如分两折，各训一个价值函数，给对方那一折打标签；
  - 更早停止，或冻结视觉编码器。
  - 先用 v1 的标签训一版策略看效果，也是可行的。
- 备份：网盘 `/root/sj-fs/so101_value_v1`（权重 1.8 GB 加标签）；本机 `runs/so101_value_v1`（不含权重）。

## 数据缺口（2026-11-11 分析）

分析方法：取每段第一次合拢夹爪的那一帧。肩部旋转为负是左侧，这一点用孪生里记录的左右核对过。

- **左侧示教并不少，但抓法不一样。**
  - 钥匙示教：左侧 67 段，右侧 31 段。
  - 钥匙测试录像的成功率：左侧 27%（49 段），右侧 42%（31 段）。
  - 合拢时腕部旋转的中位数：
    - 人的示教：钥匙左侧 −101°、右侧 −60°；玩偶左侧 −120°、右侧 −54°；
    - 孪生专家：左侧 +6°、右侧 +21°，按"活动指在远端"算出来，都在 ±96° 以内。
  - 也就是说，左侧真实示教在教旧抓法，孪生在教新抓法，而且两者的任务文本不同。
  - 用户决定：孪生的任务文本仍用 "Put the object in the white box."。
- **失败主要是在原地反复夹。** 每段合拢次数的中位数：示教 2 次，成功的测试录像 3–5 次，失败的测试录像 11–13 次。
- **起始姿态。** 真实示教都从静止姿态出发，腕部旋转约 −94°；v5 测试从抬起姿态出发。孪生 v1 的 100 段都从同一个姿态出发，腕部旋转 15°。
- **据此生成孪生 v2：** 随机起始姿态、30% 的段先故意夹空再重抓、扩大物体位置范围，共 300 段，见 real2sim2real-pitfalls.md。
  - 用户认为速度不需要和人一致，只要任务完成得准确就行。
- **还可以补的：**
  - 真机上按新抓法录 20–30 段，以左侧为主，要用户来录；
  - 孪生里加干扰物；
  - 把孪生里失败的段留下来，当价值函数的负样本。

## RECAP v2 准备（2026-11-12）

用户要求本地数据处理完后上传服务器、准备训练。新增孪生数据共 300 段：

- `so101_twin_gen_v2_512`：236 段、76458 帧，已上传；`meta/twin_expert.json` 已补传。
- `so101_twin_gen_v2b_512`：补齐的 64 段、20784 帧，等本地转换、压缩完成再上传。
- 加在 v1 后面形成 `so101_recap_v2_512`：840 段、455730 帧。原有验证集和测试集不变，新增段全部进训练集。

自动准备流程是 `scripts/recap_prepare_v2.py`，本地运行后等待 `TWIN_V2B_DONE`，上传两批数据并在服务器逐文件核对 SHA256，然后合并数据、准备价值函数输入缓存、打标签、检查标签、准备策略缓存，最后跑两步训练冒烟测试。**不启动正式策略训练，也不重新训练价值函数**。

本次本地流程运行在用户服务 `so101-recap-v2-prepare.service` 中，独立于终端和对话。用 `systemctl --user status so101-recap-v2-prepare.service` 看进程，用 `journalctl --user -u so101-recap-v2-prepare.service` 看本地接续日志；服务器阶段另有独立日志。仅在还停留于 `waiting_for_topup` 时支持 `--resume-waiting`，其余中断需先检查状态和已有产物。

- 标签复用现有价值函数 v1 的最佳权重；`configs/so101_value_v2_labels.json` 是打标签配置，不代表训出了一个新价值函数。
- 13:19 最初按旧脚本全量推理；13:43 改为 `so101vla.value --stage label --reuse-values-from runs/so101_value_v1`：复用旧的 358488 帧价值预测，只推理新增的 97242 帧。复用前核对权重、归一化、状态/动作/任务/episode、视频缓存偏移及旧缓存前缀 SHA256、成败元数据和旧预测文件 SHA256；不一致会中止。优势和各任务阈值仍在完整 v2 数据上重算，夹空/人工接管的强制标签规则不变。旧的全量重算日志保留为 `label_full_recompute_20261112.log`。
- `runs/so101_value_v2_labels/best` 链到 v1 的 `best`，`provenance.json` 记录来源和权重哈希。
- 故意夹空帧固定标为 negative，人工接管帧固定标为 positive；其他帧按合并后训练集的每任务优势分位数打标签。
- 策略配置 `configs/so101_recap_v2.json`：继续从 v5 初始化，沿用 v1 的 5000 步、batch 16 × 4、标签 dropout 0.3。
- 顶视缓存 `/dev/shm/recap_v2`，腕部缓存 `/root/sj-fs/cache_recap_v2`，两次 prepare 共用字节相同的缓存。服务器重启后顶视缓存要重建。
- 自动流程状态：本地 `runs/twin/prepare_recap_v2_local.json`，服务器 `runs/twin/prepare_recap_v2_server.json`；`ready` 才表示全部准备及冒烟测试通过，失败记为 `failed` 并保留错误。
- 服务器各阶段日志分别放在 `runs/twin/merge_recap_v2.log`、`runs/so101_value_v2_labels/{prepare,label,report}.log`、`runs/so101_recap_v2/{prepare,smoke}.log`。
- 已有缓存且标签阶段中断时，检查并停止原标签进程后，可在服务器用 `scripts/recap_prepare_v2.py --server --resume-server-label` 接续；不会重新合并或重建价值函数缓存，不覆盖已完成的标签。
- 准备结果备份到网盘 `/root/sj-fs/so101_recap_v2_preparation`；标签、配置、报告和轻量记录会拉回本机。

准备通过后的正式训练命令（自动流程不会执行）：

```bash
cd /root/sj-tmp/so101-vla-edge
.venv/bin/python -m so101vla.train --config configs/so101_recap_v2.json --stage train
```

结果（2026-11-12 14:20，服务器状态 `ready`）：

- 840 段、455730 帧，标签已生成并校验；14676 帧故意夹空全部为 negative，16246 帧人工纠正全部为 positive。
- 复用 v1 的 358488 帧价值预测，新增推理 97242 帧；原钥匙、玩偶阈值不变，object 阈值为 0.001491。
- 策略 prepare 完成：训练采样帧 407729，验证采样点 110；两路缓存哈希与价值函数 prepare 一致，视频时间戳最大误差 1.78e-6 s。
- 两步冒烟测试通过：loss 2.327、0.964，视觉编码器、connector、语言模型和 16 层动作专家的梯度均正常；峰值显存约 22.4 GB（十进制）。冒烟测试不保存训练权重。
- 准备结果已备份到网盘；正式策略训练尚未启动。

14:28 用户授权后启动正式策略训练（`runs/so101_recap_v2/train.log`，启动记录 `launch.json`）：从 v5 的 `best/model.safetensors` 初始化，全量微调；重新建立优化器，不恢复 v5 的优化器状态。初始化权重 SHA256 为 `ce5bb8ba8bafe8f34cc8bb47aab02d22400b3dd64341b15dae288afd55ce1751`，与 prepare 后冒烟测试所用的一致。保持原配置：5000 步，batch 16 × 梯度累积 4；前 200 步线性 warmup 到 5e-5，之后余弦下降到 2.5e-6，每 500 步验证并保存最佳/最近权重及恢复状态。

16:55 正常完成 5000 步。最佳 checkpoint 为第 5000 步，离线验证 normalized_action_mae 从本次第 0 步的 0.303186 降到 0.271147（约 10.6%）；玩偶为 0.295342，钥匙为 0.250984。最终训练 loss 为 0.056891，学习率为 2.5e-6。服务器模型在 `/root/sj-tmp/so101-vla-edge/runs/so101_recap_v2/best`；该结果尚未经真机闭环验证，不能据此认定成功率提高。

18:05 最佳模型完整拉回本机 `runs/so101_recap_v2/best`（含预处理器、归一化参数、RECAP 标记）。权重 1800257992 字节，SHA256 为 `2886c45c11e9c9566f490cc6b6eeb6bb302f8bf9fbfa4b8593f3a1977b13780a`，与服务器一致。三个 JSON 文件中的 VLM/tokenizer 路径已改成本机资源目录，服务器原版保存在 `server_checkpoint_metadata/`，下载记录在 `download.json`。本机离线加载与动作预测通过，默认 positive，输出 `[1, 50, 6]` 且全部有限；检查记录为 `local_load_check.json`，未连接或驱动真机。

## 还没做

- 在孪生里让模型闭环自己跑，自动得到成败。
- 真机测试：和 v5 对比，左右两侧分开统计。
