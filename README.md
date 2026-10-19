# so101-vla-edge

本科毕设项目：低成本 SO-101 机械臂上 VLA 策略（SmolVLA）的端侧（RK3588）量化部署，以闭环任务成功率作为量化效果的评价指标。

- 计划与里程碑：[docs/plan.md](docs/plan.md)
- 仿真预研（SO101 MuJoCo 抓放：脚本专家数据、基线、DAgger、目标条件化结构、评测方法）：[docs/prestudy-sim/](docs/prestudy-sim/)
- 预研产物索引：[docs/prestudy-sim/so101-artifacts.md](docs/prestudy-sim/so101-artifacts.md)

本项目于 2026-10-19 从 QVLA（HAQ 量化探索项目，分支 `experiment/so101-goal-conditioned` @ `949f701`）拆出，从干净历史开始。QVLA 保留为 HAQ 存档。

## 目录

| 路径 | 内容 |
|---|---|
| `so101vla/` | 正在使用的代码（下表） |
| `configs/` | 训练/数据配置；`configs/nexus/so101_nexus_env.json` 为 Nexus 场景参数（原随 HF 数据集 `johnsutor/MuJoCoPickAndPlace-v1` 下载） |
| `scripts/` | 分析与检查脚本 |
| `docs/` | 计划、仿真预研记录 |
| `reference/` | 原样归档、不维护的代码（RK3588 转换与板端、HAQ、量化；早期 SO101 探索；预研时的流水线脚本），见 [reference/README.md](reference/README.md) |
| `runs/`、`artifacts/` | 运行产物与模型/数据（Git 忽略）；`runs/qvla_rk3588/` 为 QVLA 中 smolvla_base 的 RK3588 转换与测速产物（`base_rk3588_speed_v1` 等，说明见 `reference/qvla_rk3588/docs/project-route.md` 末尾几节） |

## 代码（`so101vla/`）

| 模块 | QVLA 中的原文件 | 作用 |
|---|---|---|
| `paths.py` | `qvla/evaluation/so101_paths.py` | Nexus 源码路径（`SO101_NEXUS_SOURCE`） |
| `expert.py` | `so101_scripted_expert.py` | 仿真环境与读取真值的脚本专家（状态机 + IK） |
| `generate_data.py` | `generate_so101_scripted_data.py` | 生成示教数据（含特权标签） |
| `verify_data.py`、`check_labels.py` | `verify_so101_scripted_data.py`、`check_so101_privileged_labels.py` | 数据校验 |
| `train.py` | `finetune_so100_real.py` | SmolVLA 微调训练器（含 LeRobot 真机数据 prepare；`--resume`、`--stop-step`） |
| `scripted.py` | `finetune_so101_scripted.py` | 仿真数据读取、prepare、闭环评测（`--trace`） |
| `dagger.py` | `collect_so101_dagger.py` | DAgger：策略驾驶、专家接管打标签、目标标注 |
| `chain_eval.py` | `so101_chain_eval.py` | 链式长任务评测 |
| `diagnose_fit.py` | `diagnose_so101_scripted_fit.py` | 训练/验证拟合诊断 |
| `goal.py`、`goal_eval.py`、`check_goal.py` | `so101_goal.py`、`so101_goal_eval.py`、`check_so101_goal.py` | 目标条件化结构（已暂停，保留用于复现） |
| `policy.py` | `haq_offline_eval.load_policy`（仅普通 checkpoint） | 加载 SmolVLA 与处理器 |

预研文档中的路径（`qvla/evaluation/...`、`config/...`、`runs/...`）指 QVLA 中的旧位置，对应关系见上表。

## 环境

```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl SO101_NEXUS_SOURCE=/home/loser/Study/so101-nexus   # Nexus @ 7bdb671
PY=.venv/bin/python   # torch 2.7.1+cu118, lerobot 0.6.1, mujoco 3.8.1, numpy 2.2.6
$PY -m so101vla.generate_data --config configs/so101_scripted_data_v2.json
$PY -m so101vla.train --config configs/so101_scripted_full_v2.json --stage prepare
$PY -m so101vla.scripted --config configs/so101_scripted_full_v2.json --model-dir <checkpoint> --output runs/eval_x --execute 10 --stop-on-success
```

仿真预研的模型与数据已删除（2026-10-19），`runs/so101_*`、`runs/so100_*` 只保留文本结果，见 [docs/prestudy-sim/so101-artifacts.md](docs/prestudy-sim/so101-artifacts.md)。

基础模型 `artifacts/smolvla_base`（`lerobot/smolvla_base`）与 VLM 资源 `artifacts/transfer/smolvlm2_assets` 需放在 `artifacts/` 下。

`.venv` 由 QVLA 的 `.venv-haq-local` 移来，其中 libero 包的 assets 软链接已失效（本项目不用 libero）。
