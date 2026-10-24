# 常用命令

所有命令在 fish 中、项目根目录下执行。硬件细节与实测见 [hardware.md](hardware.md)。

## 1. 每次开终端

```fish
cd ~/Study/so101-vla-edge
uv activate                      # 激活 .venv（之后 python、lerobot-* 直接可用）
source scripts/so101_env.fish    # 设置 $FOLLOWER $LEADER $TOP $WRIST $CAMERAS，设备没插会提示
```

退出环境：`deactivate`。不激活也可以用 `uv run <命令>`。

## 2. 检查设备

```fish
ls /dev/serial/by-id/            # 两块舵机板（从臂 5B14111527，主臂 5B14031528）
ls /dev/v4l/by-id/               # 相机（用 *-video-index0）
lerobot-find-port                # 交互式：拔掉一块板确认是哪个端口
python scripts/camera_fps.py $TOP $WRIST                                  # 两相机同时实测帧率（默认 640x480）
python scripts/camera_fps.py --width 1280 --height 720 $TOP               # 顶视实际使用的模式，约 26.5 fps
```

设备路径一律用 by-id；`/dev/ttyACM*`、`/dev/video*` 编号重启/插拔后会变。

## 3. 标定

```fish
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower
lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader
```

- 已有标定文件时会提示：直接回车 = 沿用旧标定；输入 `c` 回车 = 重新标定。
- 流程：摆到各关节中间位置（底座朝前、夹爪半开）回车 → 依次把除 `wrist_roll` 外的每个关节转到两端极限、夹爪全开全合 → 回车保存。
- 主、从臂用同一个中间姿态。标定文件：`~/.cache/huggingface/lerobot/calibration/{robots/so_follower/follower.json,teleoperators/so_leader/leader.json}`，重标前可先备份。

## 4. 遥操作

```fish
# 无相机
lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader

# 双相机 + rerun 实时画面
lerobot-teleoperate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower --robot.cameras=$CAMERAS --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader --display_data=true
```

Ctrl+C 退出。第一次运行手放在电源附近。

## 5. 录制示教数据

用 `scripts/record.fish`（推荐，不需要先 `uv activate`/`source`）：

```fish
# 第一次：新建数据集
fish scripts/record.fish -n so101_key_pilot -t "Put the key in the white box." -e 5 -s 40
# 之后每次：追加到同一数据集（-e 是本次新录段数）
fish scripts/record.fish -n so101_key_pilot -t "Put the key in the white box." -e 10 -s 40 --resume
```

脚本每次自己 `source scripts/so101_env.fish`，串口/相机缺失或 `$CAMERAS` 为空时拒绝启动；目录已存在而没加 `--resume` 时拒绝（不会覆盖）；固定 `--dataset.push_to_hub=false`、`--display_data=true`，数据存 `artifacts/datasets/<name>`；录完检查数据集是否含 `observation.images.top` 与 `observation.images.wrist`，缺了会打印 WARNING 并以非 0 退出。参数：`-n` 数据集名，`-t` 任务描述，`-e` 段数（默认 5），`-s` 每段秒数（默认 40），`-r` 复位上限秒数（默认 600，即一直等到按 `n`），`--dry-run` 只打印完整命令不运行。视频边录边编码（`--dataset.streaming_encoding=true --dataset.encoder_threads=2`；离线以真实画面 30 Hz 送 900 帧/路测试：两路各编出 900 帧、无丢帧，收尾 0.16 s），保存一段不到 1 秒。

- `-t` 写实际做的事（训练时作为语言指令）；同一数据集续录时 `-t` 保持一致。不同任务用不同数据集名。
- 按键（焦点在运行脚本的终端；Wayland 下读终端按键）：`→`/`n` 提前结束本段，`←`/`r` 重录本段，`Esc`/`q` 停止（已录完的段会保存）。
- 流程（每段）：

  | 步骤 | 听到 | 做什么 | 按键 |
  |---|---|---|---|
  | 录制 | `Recording episode N` | 抓钥匙 → 放进盒子 → 机械臂回起始姿态 | 做完按 `n`；失败按 `r` |
  | 复位 | `Reset the environment` | 用手把物体换位置摆好（不录，从臂仍可遥操作） | 摆好按 `n`（不按会一直等，上限 `-r` 秒） |
  | 保存 | — | 边录边编码，不到 1 秒 | 不按 |

  连上机械臂后**立即**开始第一段（不等 rerun 窗口）。录满 `-e` 段后自动保存退出，最后一段之后没有复位阶段。语音需 `speech-dispatcher`、`espeak-ng`（已装）。
- 失败时机：录制中按 `r` → 进入复位，摆好按 `n` 重录同一段（推荐）。复位阶段才按 `r` → 立即结束复位并**马上**重录，所以要先摆好再按。
- 怎么结束：
  - 当前段要保留并结束 → 按 `q`：**`q` 会保存当前这一段**（不论成败）。
  - 当前段不要并结束 → 在录制阶段按 Ctrl+C（当前段未保存即丢弃，之前的段不受影响）。
  - **不要在保存阶段按 Ctrl+C**（复位结束后保存时；开启边录边编码后这一步不到 1 秒）：会留下残缺的一段（`data/chunk-*/file-NNN.parquet` 已写而 `meta/episodes` 没记录，另有 `tmp*/` 目录）。2026-10-21 `so101_key_pilot` 第 4 段即如此，已删除残留文件（该 parquet 与 `tmp*/`），之后用 LeRobotDataset 加载确认 4 段、3821 帧。
- 续录时 LeRobot 会检查相机、fps、特征与已有数据一致（`sanity_check_dataset_robot_compatibility`），相机配置不要改。
- 直接用 `lerobot-record` 时：先 `source scripts/so101_env.fish` 并 `echo $CAMERAS` 确认非空——`--robot.cameras=` 为空时 LeRobot 不报错，会录成没有图像的数据集；一定带 `--dataset.push_to_hub=false`。参数写法见 `scripts/record.fish`。
- 2026-10-21 前两次试录（`so101_pickplace_pilot` 2 段、`so101_key_pilot` 1 段）都因终端里的 `$CAMERAS` 为空（`CAMERAS` 加入 env 文件后没重新 `source`）而没有图像，已删除；之后改用 `scripts/record.fish`。
- 录制要点：主臂和人手不进顶视画面；固定光照（拉窗帘、开灯）；每段同一起始姿态；物体位置在工作区内分散；失败或犹豫的段按 `r` 重录。

## 6. 查看与回放数据

```fish
# 用 rerun 查看某一段（相机画面 + 关节曲线）
lerobot-dataset-viz --repo-id local/so101_key_pilot --root artifacts/datasets/so101_key_pilot --episode-index 0

# 让从臂重放某一段的动作（会真的动，注意周围）
lerobot-replay --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower \
  --dataset.repo_id=local/so101_key_pilot --dataset.root=artifacts/datasets/so101_key_pilot --dataset.episode=0
```

换数据集时把 `repo_id`、`root` 换成对应名字，`episode`/`episode-index` 从 0 开始。

删除指定段（段号从 0 开始；先另存为新数据集，检查后再替换，原数据集留作备份）：

```fish
lerobot-edit-dataset --repo_id local/so101_key_pilot --root artifacts/datasets/so101_key_pilot \
  --new_repo_id local/so101_key_pilot --new_root artifacts/datasets/so101_key_pilot_clean \
  --operation.type delete_episodes --operation.episode_indices "[19, 20]"
uv run python scripts/check_dataset.py artifacts/datasets/so101_key_pilot_clean    # 段数、帧数对得上再替换
mv artifacts/datasets/so101_key_pilot artifacts/datasets/so101_key_pilot_backup
mv artifacts/datasets/so101_key_pilot_clean artifacts/datasets/so101_key_pilot
```

删除后后面的段号会前移。2026-10-21 `so101_key_pilot` 原第 19、20 段（10–12 s，机械臂没动，第 20 段顶视里有人手在摆物体：复位阶段多按了一次 `n`，把摆放录成了示教）即这样删除，26 段 → 24 段、19525 → 18864 帧，原数据集备份为 `so101_key_pilot_backup_26ep`。

2026-10-22 补录 50 段（第 49–98 段，两次各 25 段）后，删除原第 91 段：用户回忆有一段钥匙放到盒子边上没放进去、误按了 `n`；末帧拼图里只有这一段白盒是空的，钥匙落在白盒和玩偶盒后方桌面（第 49–98 段其余各段末帧钥匙均在白盒内）。99 段 → 98 段、60261 → 59591 帧（该段 670 帧，22.3 s），原数据集备份为 `so101_key_pilot_backup_99ep`。第 0–48 段（实验一所用）段号不变。

检查数据质量（每段时长、时间戳、视频帧数、运动中的重复帧比例、关节/动作范围；`--sheet` 存每段首/中/末帧拼图）。`moving 0%` 的段机械臂没动，多半是误录，应删除：

```fish
uv run python scripts/check_dataset.py artifacts/datasets/so101_key_pilot --sheet /tmp/key_pilot_sheet.jpg
```

## 7. 训练与评测

训练（`so101vla/train.py`；每个实验一个记录，见 [experiments/](experiments/)，如 [实验一](experiments/exp1-key-v1.md)）：

```fish
uv run python -m so101vla.train --config configs/so101_key_v1.json --stage prepare   # 解码视频、算训练段统计、写缓存
uv run python -m so101vla.train --config configs/so101_key_v1.json --stage train     # 加 --resume 从 last/ 接着训练
tail -f runs/so101_key_v1/train.log                                                  # validation 行是验证误差
```

8 GB 显存很紧（训练约 7 GB）：浏览器、VS Code 也会占几百 MB 显存，训练启动报 `CUDA out of memory` 时，在命令前加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（fish 写作 `env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run ...`），或关掉这些程序。torch.compile 在本机不提速，见 [notes/training-speed.md](notes/training-speed.md)。

真机测试（`scripts/eval.fish`，不需要先 `uv activate`/`source`）：

```fish
# 第一次：新建评测数据集（名字必须以 rollout_ 开头）
fish scripts/eval.fish -n rollout_key_v1_best -t "Put the key in the white box." -p runs/so101_key_v1/best -e 10 -s 40
# 继续测（追加到同一评测集）
fish scripts/eval.fish -n rollout_key_v1_best -t "Put the key in the white box." -p runs/so101_key_v1/best -e 10 -s 40 --resume
```

- 不接主臂：脚本启动时从臂所处姿态就是"起始姿态"，每段结束后从臂 1 秒内回到该姿态并保持。所以**启动前把从臂摆到录示教时的起始姿态**。
- 流程（每段）：听到 `Recording episode N` → 模型控制从臂（首次推理约 0.6 s，之后每 50 步/1.7 s 停顿约 0.17 s 推理）→ 成功或明显失败后按 `n` → 从臂回起始姿态 → 摆好物体按 `n` 开始下一段。`r` 重测本段（该段不保存），`q` 停止。
- 异步推理加 `--rtc`（LeRobot Real-Time Chunking）：后台线程在队列剩 30 步时就生成下一块，并引导新块前 20 步与旧块衔接（实测推理延迟约 11 步，视野要大于它）。机械臂不再每 1.7 s 停一下。2026-10-24 起，`so101vla/rollout.py` 修复了 LeRobot RTC 的两个问题：重置后用旧观测推理；新动作块开头按估算的推理耗时多丢约 11 步，现在改为按实际执行的步数丢弃，因此原来的 `Indexes diff is not equal to real delay` 警告不再出现。详见 [notes/inference-sync-vs-rtc.md](notes/inference-sync-vs-rtc.md)。离线检查和第 6500 步模型的真机对比见 [notes/inference-sync-vs-rtc.md](notes/inference-sync-vs-rtc.md)（视野 10 已测；视野 20 的真机效果未测量）。CSV 的 note 列记录 `sync`/`rtc`。同步与 RTC 的结果用不同的评测名分开，例如 `rollout_pilot_v2_s6500_doll_rtc`。
- 只试一次、不保存（`scripts/try.fish`，LeRobot `--strategy.type=base`）：不建评测数据集、不写 CSV。`-s` 秒后（默认 40，`-s 0` 一直跑）或按 Ctrl+C 停止，从臂回到启动时的姿态后退出。`n`/`r` 键无效。
  ```fish
  fish scripts/try.fish -t "Put the doll in the white box." --rtc          # 默认 runs/so101_pilot_v2/best
  fish scripts/try.fish -t "Put the key in the white box." --rtc -s 0      # 一直跑，Ctrl+C 停
  ```
- 平滑动作（`eval.fish` 和 `try.fish` 自 2026-10-23 起**默认开启**，σ=2；`--smooth 0` 关闭，`--smooth 3` 更平滑）：对模型每次输出的 50 步动作块沿时间做高斯平滑，减少抖动，不加延迟。分析和离线结果见 [notes/action-smoothing.md](notes/action-smoothing.md)；真机实测：下发动作每秒反转从约 6 次降到约 0.04 次，3 Hz 以上的能量占比从 31–52% 降到约 4%（示教为 3%）。CSV 的 note 列会记成 `rtc+smooth2` 或 `sync+smooth2`。
  - 量化比较两组录像（反转/秒、≥3 Hz 能量占比、jerk、高频 RMS，逐关节）：`uv run --frozen python scripts/jitter_analysis.py artifacts/datasets/<不平滑> artifacts/datasets/<平滑>`；对比图：`uv run --frozen --with matplotlib python scripts/smoothing_plot.py`（数据集名写在脚本开头）。
- **安全**：第一次手放在电源旁；从臂乱动、顶桌面或卡住时立刻按 `q`，或 Ctrl+C，或断电。
- 结束后脚本逐段问是否成功（`y`/`n`/`s` 跳过不计），追加到 `runs/real_eval/<name>.csv` 并打印成功率；评测录像存在 `artifacts/datasets/<name>`，可用 §6 的 `lerobot-dataset-viz` 回看。
- 训练用的任务文本、相机配置要与录制一致；物体摆放用训练中出现过的范围，另行记录摆放位置。

### 纠错数据（人接管纠正）

`scripts/dagger.fish`：模型控制从臂完成一次任务尝试（一段），出错时人用主臂接管纠正，纠正完再交还模型。每帧记录 `intervention`（True = 主臂在控制）。设计与验证见 [notes/correction-collection.md](notes/correction-collection.md)（目前只在假硬件上验证过）。

```fish
fish scripts/dagger.fish -n rollout_dagger_doll_r1 -t "Put the doll in the white box." --rtc            # 默认 runs/so101_pilot_v2/best，10 段，不限时（按 n 结束），平滑 σ=2
fish scripts/dagger.fish -n rollout_dagger_doll_r1 -t "Put the doll in the white box." --rtc --resume   # 追加
```

- 需要接主臂。启动前把两只臂都摆到录示教时的起始姿态。终端窗口要保持在前台（Wayland 下按键从终端读）。
- 启动后和每段之间都先等待：摆好物体后按 `n` 才开始录下一段，摆放过程不会被录进去。
- 按键：
  - `空格`：模型 → 暂停。模型停止，主臂在 2 s 内被带到从臂的姿态并保持住。这时握住主臂。
  - `Tab`：暂停 → 纠正。主臂卸力，从臂跟随主臂。
  - 纠正中按 `空格`：交还模型。主臂停在你松手的位置并保持，不会掉下来。纠正中按 `Tab` 则两只臂都保持不动（暂停）。
  - `n` 结束本段并保存。按之前先松开主臂，它会被带回起始姿态后卸力。`r` 重来本段（不保存），`q` 停止。
- 规则：一旦接管，至少完成当前这一小步（例如夹稳并抬起）再交还。本段里模型在同一个地方又失败时，由人做完本段再按 `n`。暂停期间的帧不记录。
- 每段默认不限时，只有按 `n` 才结束。`-s 秒数` 可以设上限，但只按实际录下的时长计算（暂停和主臂对齐的时间不算），而且不会在纠正过程中截断。
- 结束后逐段显示帧数、接管次数、人控帧占比、是否由人做完，并询问成功与否（`y` = 任务完成，有没有人帮忙都算）。结果追加到 `runs/dagger/<name>.csv`，最后打印"完成数 / 无接管完成数 / 总接管次数"。随时可查看统计：`uv run --frozen python -m so101vla.correction artifacts/datasets/<name>`。

合并成训练集（示教 + 纠错段）：

```fish
uv run --frozen python scripts/merge_corrections.py --demo artifacts/datasets/so101_pilot_v2 \
    --demo-partition configs/so101_pilot_v2_partition.json --out so101_dagger_r1 \
    --partition-out configs/so101_dagger_r1_partition.json artifacts/datasets/rollout_dagger_doll_r1 artifacts/datasets/rollout_dagger_key_r1
```

- 示教集要先加上全为 True 的 `intervention` 列才能和纠错段合并。脚本会把带这一列的副本缓存在 `artifacts/datasets/so101_pilot_v2_intervention`（3.6 GB，已建好），以后合并直接复用。
- 合并后编号：示教在前，编号不变，验证段沿用原来的；纠错段全部归入训练。训练配置复制 `configs/so101_pilot_v2.json`，改 `dataset_root`、`partition`、`output`。`train.py` 遇到 `intervention` 列时只用人控帧训练，每帧的动作块在所在纠正片段结束处截止，之后补 padding。
- 不接硬件的完整自测（假从臂、主臂和推理 → 录制 → 合并 → prepare）：`uv run --frozen python scripts/correction_selftest.py`，约 3 分钟。

## 8. 环境管理（uv）

```fish
uv sync                          # 按 uv.lock 建立/更新 .venv
uv add <包>                      # 加依赖（同时更新 pyproject.toml 与 uv.lock）
uv lock --upgrade-package <包>   # 升级单个包
uv tool install <包>             # 全局命令行工具（如 huggingface_hub 提供 hf）
uv cache prune                   # 清理缓存中不再使用的包
```

## 9. 常见报错

| 报错 | 原因与处理 |
|---|---|
| `Missing required field(s) port` | 新终端里没 `source scripts/so101_env.fish`，`$LEADER`/`$FOLLOWER` 为空 |
| `failed to set fps=30 (actual_fps=31.0)` | 顶视 1280×720 模式驱动只给 31 fps，相机配置写 `fps: 31`（`$CAMERAS` 已是） |
| `torchcodec ... Could not load libtorchcodec` 警告 | 系统 FFmpeg 9 不被 torchcodec 0.11 支持，LeRobot 自动改用 pyav，可忽略 |
| 串口 `Permission denied` | 用户需在 `uucp` 组（已在） |
| `OpenCVCamera(...) read thread is not running.`（测试/录制中途崩溃） | 相机在 USB 上掉线了（`journalctl -k \| grep 'USB disconnect'` 可查）。2026-10-23 腕部相机（ZSKJ，接在 USB hub 上）8 分钟内掉线 3 次，之前录 200 段时一次没掉过。先把腕部相机线两头重新插紧、腕部转动处留够余量并固定；还掉就直接插电脑 USB 口，不经过 hub。崩溃前已完成的段会保存，正在录的那段也会被截断保存为最后一段；脚本问成功与否时，最后这段按 `s` 跳过 |
| 某相机/串口找不到 | `ls /dev/serial/by-id/ /dev/v4l/by-id/` 检查是否插好；`source scripts/so101_env.fish` 会提示缺哪个 |
