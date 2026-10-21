# 真机硬件记录

## SO-101（2026-10-21 到货）

| 部件 | 设备路径（用 by-id，`/dev/ttyACM*`、`/dev/video*` 编号重启/插拔后会变） | 实测 |
|---|---|---|
| 舵机板 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14111527-if00`、`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031528-if00`（哪块是主/从臂以 `lerobot-find-port` 为准） | 两块板 ID 1–6 均应答，型号 777（STS3215），1 Mbps；出厂已设 ID，无需 `lerobot-setup-motors` |
| 顶视相机 | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_2.0_Camera-video-index0`（USB `0c45:6368`） | 见下；画面相对桌面旋转约 90° |
| 腕部相机 | `/dev/v4l/by-id/usb-ZSKJ-260319-K_Integrated_Webcam_HD-video-index0`（USB `0c45:64ab`） | 640×480 MJPG 实测 30.0 fps；有 `auto_exposure` / `exposure_time_absolute` 控制 |

标定：`lerobot-calibrate`，id 为 `follower` / `leader`，文件在 `~/.cache/huggingface/lerobot/calibration/`。

## 顶视相机帧率（2026-10-21）

请求 30 fps，`scripts/camera_fps.py`（OpenCV，按墙钟计时）与 `v4l2-ctl --stream-mmap --verbose`（按缓冲时间戳间隔）结果一致：

| MJPG 分辨率 | 帧间隔 | 帧率 |
|---|---:|---:|
| 1280×720 | 37.7 ms | 26.5 |
| 1920×1080、1280×960、1024×768、960×720、800×600、640×480、640×360、432×240、352×288、320×240 | 50.4 ms | 19.8 |

- 不是 OpenCV 问题：只 `grab()` 不解码、改设置顺序、关颜色转换均为 19.9 fps；`v4l2-ctl` 直接取流按墙钟也约 20 fps（其打印的 “30.000 fps” 来自时间戳估计，不可信）。
- 不是驱动丢帧：缓冲序号连续，`uvcvideo nodrop=1`。
- 不是带宽：640×480 MJPG（约 65 KB/帧，1.3 MB/s）与 YUYV（614 KB/帧，12.2 MB/s）帧间隔相同；720p 吞吐 5.15 MB/s 反而更快。
- 不是曝光/工频：相机未提供曝光控制；`power_line_frequency` 50/60 Hz/关闭结果相同；用户实测很暗与正常光照下均约 20 fps。

结论：帧时序由相机固件按分辨率模式固定，软件无法提高到 30 fps。

固件方向的探查（只读）：主控为 Sonix，固件版本 27.02，扩展单元 GUID `28f03370-6311-4a2e-ba2c-6890eb334016`（Sonix SYS_HW_CTRL，unit 3，selector 1–5，GET_LEN 4/8/11/11/11）。按开源 SONiX_UVC_TestAP 的 ASIC 读协议（selector 1：SET_CUR `[addrLo, addrHi, 0, 0xFF]` 后 GET_CUR，值在 byte 2）读到芯片 ID 寄存器 0x101f = 0x71，不是该工具支持的 SN9C291（0x90）/SN9C292（0x92），公开资料中查不到其寄存器表；读取后相机工作正常（仍 19.9 fps）。继续需要对无文档的 I2C/时序寄存器盲写（易失，但效果与风险都无法确认），未进行；selector 3（参考芯片上为 Flash 控制）从未访问。

对录制与部署的影响：LeRobot 0.6.1 的 `so101_follower.get_observation()` 用 `read_latest()`（不阻塞，取后台线程最新帧），所以 20 fps 相机不会拖慢 30 Hz 控制/录制循环；代价是按 30 Hz 存的顶视帧约 1/3 与上一帧重复，图像最旧约 50 ms（30 fps 相机约 33 ms）。部署时策略只在每个动作块开始时取图，20 fps 足够。训练与部署须用同一相机与设置。
