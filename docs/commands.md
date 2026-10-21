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

脚本每次自己 `source scripts/so101_env.fish`，串口/相机缺失或 `$CAMERAS` 为空时拒绝启动；目录已存在而没加 `--resume` 时拒绝（不会覆盖）；固定 `--dataset.push_to_hub=false`、`--display_data=true`，数据存 `artifacts/datasets/<name>`；录完检查数据集是否含 `observation.images.top` 与 `observation.images.wrist`，缺了会打印 WARNING 并以非 0 退出。参数：`-n` 数据集名，`-t` 任务描述，`-e` 段数（默认 5），`-s` 每段秒数（默认 40），`-r` 复位秒数（默认 15）。

- `-t` 写实际做的事（训练时作为语言指令）；同一数据集续录时 `-t` 保持一致。不同任务用不同数据集名。
- 按键（焦点在运行脚本的终端；Wayland 下读终端按键）：`→`/`n` 提前结束本段，`←`/`r` 重录本段，`Esc`/`q` 停止（已录完的段会保存）。
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

检查数据质量（每段时长、时间戳、视频帧数、运动中的重复帧比例、关节/动作范围；`--sheet` 存每段首/中/末帧拼图）：

```fish
uv run python scripts/check_dataset.py artifacts/datasets/so101_key_pilot --sheet /tmp/key_pilot_sheet.jpg
```

## 7. 训练与评测

真机数据的训练配置待定（基于 `so101vla/train.py`，prepare 阶段读取 LeRobot 数据集）；写好后补充到这里。

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
| 某相机/串口找不到 | `ls /dev/serial/by-id/ /dev/v4l/by-id/` 检查是否插好；`source scripts/so101_env.fish` 会提示缺哪个 |
