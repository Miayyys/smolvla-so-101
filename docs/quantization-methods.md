# 量化方法候选：尽量降低 INT8 损失（2026-11-04 整理，未实施、未测量）

这份笔记是 RK3588 部署前的方法梳理，只列出方法和可行性判断，不是方案。路线决定见 [plan.md](plan.md)"路线决定：2026-11-04"。

**前提**：RK3588 的 NPU 主要支持 INT8（w8a8，权重和激活都是 8 位）和 FP16；其他位宽是否可用，要用 RKNN-Toolkit2 实测确认。之前已确认 RK3588 不支持稀疏推理（`reference/qvla_rk3588/docs/project-route.md`）。RKNN 采用静态量化：激活值的缩放系数由校准数据事先确定，推理时不再动态计算。因此重点是怎样让 INT8 的损失尽量小，以及哪些层保留 FP16。

**可用性标记**：✅ 可以直接用；⚠️ 有前提条件，需要先确认工具链是否支持；❌ 不适合我们的场景。表中提到的 RKNN 选项和工具（`quantized_algorithm`、`quantized_method`、`accuracy_analysis`、混合量化、导入 QDQ 模型）都是按记忆整理的，用之前要对照 RKNN-Toolkit2 的文档和实测确认。

## 一、校准方法：决定激活值的量化范围

| 方法 | 原理 | 可用性 |
|---|---|---|
| MinMax | 直接取校准数据的最小值和最大值作为范围 | ✅ RKNN `normal`；容易被离群值拖累 |
| KL 散度 | 选截断范围，让量化前后的分布差异（KL 散度）最小 | ✅ RKNN `kl_divergence` |
| MSE / MMSE | 选截断范围，让量化误差的均方值最小 | ✅ RKNN `mmse` |
| 按通道量化权重 | 每个输出通道单独一个缩放系数 | ✅ RKNN `quantized_method='channel'`，应默认使用 |
| 分时间步校准（Q-Diffusion） | 扩散/流匹配模型每个去噪步的激活分布不同，校准数据要覆盖所有步 | ✅ 很适合：动作专家循环 10 步，收集校准数据时 10 步都要采 |

## 二、等价变换：转换前改权重，计算图不变

这类方法在 PyTorch 里改好权重再导出，RKNN 看到的仍是普通模型，所以容易落地。

| 方法 | 原理 | 可用性 |
|---|---|---|
| SmoothQuant | 激活值按通道除以一个系数、下一层权重乘以同一个系数，把激活的离群值"挪"到权重上；系数吸收进前面的 Norm 层 | ✅ 适合语言部分和动作专家 |
| CLE 跨层均衡（DFQ） | 调整相邻两层权重的通道尺度，让各通道范围接近 | ✅ 原本多用于 CNN |
| Bias Correction（DFQ） | 补偿量化引起的输出均值偏移 | ✅ 只改 bias |
| RepQ-ViT | ViT 中 LayerNorm 之后的激活按通道差异很大；把按通道的缩放重参数化进 LayerNorm 和下一层 | ✅ 适合视觉编码器（SigLIP 是 ViT） |
| OmniQuant | 等价变换的系数和权重截断范围用少量数据学出来 | ⚠️ 学出的截断范围要能交给 RKNN 使用 |
| QuaRot / SpinQuant | 用旋转（Hadamard 等正交变换）消除离群值 | ⚠️ 能合并进权重的部分可用；需要在线旋转的部分会多出运算，不适合 |

## 三、权重取整优化：逐层/逐块重建

共同前提：RKNN 要能直接使用我们给出的量化权重和缩放系数（例如导入带量化节点的 QDQ ONNX），而不是自己重新量化。这一点需要先实测确认。

| 方法 | 原理 | 可用性 |
|---|---|---|
| AdaRound | 用少量数据学每个权重向上还是向下取整，让该层输出误差最小 | ⚠️ |
| BRECQ | 以块（例如一个 Transformer 块）为单位重建，考虑层间相互影响 | ⚠️ 通常比 AdaRound 好 |
| QDrop | 重建时随机丢弃部分激活量化，让结果更稳定 | ⚠️ |
| GPTQ | 逐列量化权重，用未量化的列补偿误差，补偿量由输入的二阶统计 XXᵀ 决定；由 OBQ 发展而来 | ⚠️ 优势主要在 3–4 位；8 位时直接四舍五入已经几乎无损，收益很小 |
| AWQ | 按激活的重要性保护少数权重通道，只量化权重 | ❌ 偏向 4 位权重量化；如果语言部分改用 RKLLM 的低位宽再考虑 |
| PTQ4ViT | softmax/GELU 之后的激活用两段不同尺度量化 | ❌ 需要专门算子，RKNN 不支持 |

## 四、混合精度：决定哪些层保留 FP16

| 方法 | 原理 | 可用性 |
|---|---|---|
| 逐层敏感度分析 | 一次只量化一层，看输出误差 | ✅ RKNN `accuracy_analysis` |
| RKNN 混合量化 | 指定的层保留 FP16，其余 INT8 | ✅ 内置 |
| HAWQ | 用 Hessian 信息衡量各层敏感度，据此分配位宽 | ✅ 可用来决定 FP16 的层 |
| HAQ | 用强化学习搜索每层位宽，以实测延迟为约束 | ✅ QVLA 中做过（`reference/qvla_rk3588/haq/`），结论不能直接迁移，思路可复用 |

SmolVLA 中可能比较敏感的位置（推测，待 `accuracy_analysis` 验证）：动作专家的输入输出投影和时间嵌入（直接决定动作数值）；Norm、softmax 附近；专家循环 10 次，误差会累积，可能比视觉部分更敏感。

## 五、QAT：量化感知训练

| 方法 | 原理 | 可用性 |
|---|---|---|
| LSQ / LSQ+ | 训练时模拟量化，缩放系数（LSQ+ 还有偏移）也作为参数学习 | ✅ 需要服务器训练；训练时模拟的量化方式必须和 RKNN 实际执行的完全一致 |
| PACT | 激活的截断上限作为参数一起学 | ✅ 同上 |
| QAT + 蒸馏 | 浮点模型当老师，量化模型学习老师的动作输出 | ✅ 与 LSQ 配合 |
| EfficientQAT | 先逐块训练，再端到端只训练缩放系数，降低成本 | ⚠️ 主要针对大模型低位宽 |

## 六、和量化相关的结构改动

QVLA 中对 `smolvla_base` 的板端实测：512 分辨率混合 INT8 约 2.9 s，图像 256 加完整 16 层专家约 1.5 s，图像 128 加 4 层专家约 0.6 s。30Hz、每块 50 步的时间预算约 1.67 s。降低分辨率、减少去噪步数、精简专家层数带来的精度损失，可能比量化本身更大，**必须重新训练恢复**，需要和量化一起规划。

## 评价方式

- 离线：同一批真机画面，比较量化模型和浮点模型的动作输出误差；可以逐个模块（视觉、语言、专家）单独量化来比较。
- 最终：闭环成功率。实验三、实验四都出现过离线误差和真机结果不对应的情况。

## 候选组合（顺序建议，未定）

1. 只用 RKNN 内置功能：按通道权重 + MMSE/KL 校准 + 真机画面、覆盖 10 个去噪步的校准集 → `accuracy_analysis` 找敏感层 → 敏感层 FP16。
2. 转换前改权重：语言部分和专家用 SmoothQuant，视觉编码器用 RepQ-ViT，加 Bias Correction。
3. 损失仍大时：如果 RKNN 能导入自定义量化参数，试 AdaRound/BRECQ；否则做 LSQ QAT 加蒸馏。
4. 可能的创新点：以闭环成功率和实测延迟为约束的混合精度搜索（plan.md"待定"一节）。

## 出处

arXiv 编号是凭记忆写的，没有逐个核对。引用到开题报告或论文之前要核对，核对后的条目移到 [references.md](references.md)。

| 方法 | 论文 |
|---|---|
| KL 校准 | Migacz, *8-bit Inference with TensorRT*, GTC 2017（NVIDIA 报告） |
| DFQ（CLE、Bias Correction） | Nagel et al., *Data-Free Quantization Through Weight Equalization and Bias Correction*, ICCV 2019, arXiv 1906.04721 |
| AdaRound | Nagel et al., *Up or Down? Adaptive Rounding for Post-Training Quantization*, ICML 2020, arXiv 2004.10568 |
| BRECQ | Li et al., *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction*, ICLR 2021, arXiv 2102.05426 |
| QDrop | Wei et al., *QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization*, ICLR 2022, arXiv 2203.05740 |
| OBQ | Frantar & Alistarh, *Optimal Brain Compression*, NeurIPS 2022, arXiv 2208.11580 |
| GPTQ | Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023, arXiv 2210.17323 |
| SmoothQuant | Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML 2023, arXiv 2211.10438 |
| AWQ | Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024, arXiv 2306.00978 |
| OmniQuant | Shao et al., *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models*, ICLR 2024, arXiv 2308.13137 |
| QuaRot | Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, 2024, arXiv 2404.00456 |
| SpinQuant | Liu et al., *SpinQuant: LLM Quantization with Learned Rotations*, 2024, arXiv 2405.16406 |
| PTQ4ViT | Yuan et al., *PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization*, ECCV 2022, arXiv 2111.12293 |
| RepQ-ViT | Li et al., *RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers*, ICCV 2023, arXiv 2212.08254 |
| Q-Diffusion | Li et al., *Q-Diffusion: Quantizing Diffusion Models*, ICCV 2023, arXiv 2302.04304 |
| HAQ | Wang et al., *HAQ: Hardware-Aware Automated Quantization with Mixed Precision*, CVPR 2019, arXiv 1811.08886 |
| HAWQ | Dong et al., *HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision*, ICCV 2019, arXiv 1905.03696 |
| PACT | Choi et al., *PACT: Parameterized Clipping Activation for Quantized Neural Networks*, 2018, arXiv 1805.06085 |
| LSQ | Esser et al., *Learned Step Size Quantization*, ICLR 2020, arXiv 1902.08153 |
| LSQ+ | Bhalgat et al., *LSQ+: Improving Low-bit Quantization through Learnable Offsets and Better Initialization*, CVPR Workshops 2020, arXiv 2004.09576 |
| EfficientQAT | Chen et al., *EfficientQAT: Efficient Quantization-Aware Training for Large Language Models*, 2024, arXiv 2407.11062 |

VLA 专门的量化工作较新，还没有检索，暂不列出。
