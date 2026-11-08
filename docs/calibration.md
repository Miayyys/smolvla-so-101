# 相机与机械臂标定（2026-11-08）

real2sim2real 第 0 步的记录。用平板显示 ChArUco 板完成，脚本在 `scripts/calib_*.py`，图片和原始数据在 `runs/calib/`、`artifacts/calibration/`（Git 忽略），所以下面把结果数值也写全。

**结论**：顶视内参、顶视外参、腕部内参、腕部手眼都已完成，可以进入下一步。两路内参很准；外参和手眼的误差约 5 mm，主要来自机械臂本身，够第一版孪生使用。

**有效期**：相机位置变了要重做外参；腕部相机拆装或碰歪要重做手眼；底座挪动要重做外参。内参只对标定时的分辨率有效，改变镜头焦距也要重做。

## 结果

| 项目 | 结果文件 | 主要指标 |
|---|---|---|
| 顶视内参 | `runs/calib/top/intrinsics_standard.json` | RMS 0.214 px，留出 0.278 px，30 张全部使用 |
| 顶视外参 | `runs/calib/top_extrinsics_2/extrinsics.json` | 拟合平均 4.2 mm、最大 7.8 mm；留出一个位置平均 4.7 mm、最大 8.7 mm |
| 腕部内参 | `runs/calib/wrist/intrinsics_standard.json` | RMS 0.171 px，留出 0.234 px，30 张全部使用 |
| 腕部手眼 | `runs/calib/wrist/handeye.json` | 重投影 RMS 13.3 px，留出 11.6 px；板子位姿和顶视相机差 3.3 mm、2.9° |

### 内参（OpenCV 标准模型，畸变顺序 k1 k2 p1 p2 k3）

| | 分辨率 | fx | fy | cx | cy | 畸变 |
|---|---|---|---|---|---|---|
| 顶视 | 1280×720 | 647.278 | 647.668 | 702.112 | 274.054 | 0.00801, −0.011118, 0.000206, −0.000178, 0.001918 |
| 腕部 | 640×480 | 456.002 | 455.984 | 326.927 | 254.664 | −0.032941, 0.066652, 0.000387, 0.000667, −0.038946 |

- 顶视视场约 89°×58°，畸变几乎为零，可以按针孔模型渲染。
- 顶视主点偏离画面中心约 (+62, −86) px，是真实的：奇偶分半结果一致，强制放在中心后 RMS 从 0.21 涨到 1.33 px。渲染必须用实测主点。
- 腕部视场约 70°×56°，畸变不能忽略。画面下方被夹爪挡住，覆盖图有 3 格没有角点，那里的畸变是外推的。
- `intrinsics_rational.json` 不要用：误差没有改善，系数大到几十到近 200，属于过拟合。

### 顶视外参

`P_cam = R · P_base + t`，单位米，基座是 MJCF 的 `base` 坐标系。

```
R = [[-0.000681, -0.999705, -0.024289],
     [-0.999807,  0.000203,  0.019656],
     [-0.019645,  0.024298, -0.999512]]
t = [-0.0148, 0.26911, 0.43182]
```

相机在基座坐标系中的位置约 (277.5, −25.3, 426.0) mm。MJCF 为 Nexus @ `7bdb671` 的 `so101_new_calib.xml`。

### 关节零位

`q_mjcf_deg = q_lerobot_deg + offset_deg`，`q_lerobot_deg` 就是数据集里存的角度。

| 关节 | 碰靶点估计（已采用） | 腕部图像估计（独立，仅作验证） |
|---|---|---|
| shoulder_pan | 0.01° | 0.28° |
| shoulder_lift | −2.21° | −4.77° |
| elbow_flex | −6.43° | −8.41° |
| wrist_flex | −2.81° | 1.32° |
| wrist_roll | 0.83° | 0.66° |

- 这些数必须用：不修正零位时，外参拟合平均误差是 27 mm。
- 不要当成精确的物理量。肘部和腕部互相耦合，换一批数据就有几度变化。
- 肩部旋转的零位和基座相对相机的朝向无法区分，算在了基座朝向里，投影和孪生不受影响。
- **独立验证**：只用腕部图像估计的零位，去预测碰靶点数据，平均误差 5.2 mm（留出一个位置 5.9 mm）。两批独立数据结论一致。

### 腕部手眼

`T_gripper_cam`：OpenCV 相机坐标系（x 右、y 下、z 向前）相对 MJCF `gripper` 刚体，单位米。

```
[[ 0.998976,  0.028168, -0.035408, 0.004935],
 [ 0.010713, -0.907575, -0.419753, 0.059015],
 [-0.043959,  0.418944, -0.906947, 0.005782],
 [ 0,         0,         0,        1       ]]
```

- 相机在夹爪坐标系中约 (4.9, 59.0, 5.8) mm，和 MJCF 默认 `wrist_cam`（0, 40, −40）差别较大，默认值不能用。
- 13 px 约相当于 5 mm，主要来自机械臂运动学误差（推测为舵机回差和结构形变）。把关节零位也一起优化后只降到 12.9 px，所以重新采集意义不大。
- 手眼沿夹爪轴方向的位置随所用零位变化约 10 mm（零位来自腕部图像时 z 约为 −4 mm），要按 ±5 mm 的不确定理解。
- 第 2 步验证孪生时，腕部渲染和实拍预计有 10–15 px 错位，靠近物体时要重点检查。

## 方法

**标定板**：平板（2880×1800）全屏显示，每个黑格实测约 28 mm，7 格总宽软尺量得 19.5 cm。解算用 27.9 mm；27.86–28.0 mm 之间外参结果几乎不变。图片是 `cv2.aruco.CharucoBoard((7, 5), 340, 256, DICT_6X6_250)`，格子 340 px，码 256 px，板面 2380×1700 px 居中在白底上（左边距 250、上边距 50）。带靶点的版本在左右白边加了十字靶：A (125, 300)、B (2755, 300)、C (2755, 1500)、D (125, 1500)。

**顶视内参**：相机不动，移动平板，脚本在板子稳住且姿态够新时自动保存。

```fish
source scripts/so101_env.fish
.venv/bin/python scripts/calib_intrinsics.py capture --camera $TOP --width 1280 --height 720 --fps 31 --out runs/calib/top --preview
.venv/bin/python scripts/calib_intrinsics.py solve --views runs/calib/top --square-mm 27.9
```

**顶视外参**：平板平放，顶视相机得到板面位姿，所以每个十字靶在相机坐标系中的位置已知。手托机械臂，用固定爪尖碰靶点，由正运动学得到同一点在基座坐标系的位置，稳住 2 秒自动记录。第二轮采了 5 个位置，第 1 组采集时平板被碰动，解算时用 `--skip 1` 排除。全程只读舵机，不写任何寄存器，力矩保持关闭。

```fish
.venv/bin/python scripts/calib_top_extrinsics.py collect --camera $TOP --port $FOLLOWER --intrinsics runs/calib/top/intrinsics_standard.json --out runs/calib/top_extrinsics_2
.venv/bin/python scripts/calib_top_extrinsics.py solve --out runs/calib/top_extrinsics_2 --square-mm 27.9 --skip 1
.venv/bin/python scripts/calib_top_extrinsics.py check --camera $TOP --port $FOLLOWER --out runs/calib/top_extrinsics_2
```

`check` 把 MJCF 机械臂投影到实拍画面，结果存为 `overlay.jpg`，这次两个爪子和机身都对得上。

**腕部内参和手眼**：平板不动，手动移动机械臂，腕部旋转覆盖 −164° 到 +55°。开始和结束时顶视相机各看一次平板，确认没被碰动（这次移动 0.09 px）。

```fish
.venv/bin/python scripts/calib_wrist.py capture --wrist $WRIST --top $TOP --port $FOLLOWER --out runs/calib/wrist
.venv/bin/python scripts/calib_wrist.py solve --out runs/calib/wrist --square-mm 27.9
```

三个解算都先用已知真值的合成数据验证过：内参误差在 1.5 px 以内；碰触误差 1 mm 时外参留出误差 1.7 mm；手眼位置误差 0.6 mm、旋转 0.3°。

## 过程记录

**第一轮外参作废（`runs/calib/top_extrinsics/`，不要用）**：拟合误差 34 mm，算出的关节偏差达 47°。查出顶视相机在采集期间动过：画面相对 10 月以来所有数据（偏差约 5 px 以内）移动了约 42 px / 25 px，约 3.5 cm。用户当时在排查相机问题，之后还调整过位置。第二轮前拧紧了相机、要求固定底座，结果恢复正常。

**相机位置和旧数据不同**：现在的顶视视角和示教、v5 测试时不一样，也不打算调回去。之前的真机成绩都是旧视角下的，后续对比要在同一相机位置下重测基线。

**LeRobot 的角度零位**：LeRobot 的 DEGREES 以标定范围的中点为零，和 MJCF 零位不同，所以读原始编码器值，再按 LeRobot 的公式换算，保持和数据集一致，差异由上面的零位偏差修正。

**排除过的解释**：关节方向翻转（最优仍是全部不翻转）、`so101_old_calib.xml`（误差 83–90 mm）、以 2047.5 为零（更差）、接触点侧向偏 37 mm（留出误差 6.5 mm，比零位修正的 4.7 mm 差，且两个爪子上都没有这样的位置）。

**用屏幕显示标定板要注意**：不要把亮度开太高，这台顶视相机不能调曝光，房间暗时白格会过曝发光；平板稍微倾斜可以避开灯和相机自身的反光。

**终端交互**：预览用的 mpv 必须加 `--no-terminal`，否则它会打开 `/dev/tty` 抢走键盘输入，表现为按回车没反应。
