# 昇腾 8 卡训练加速实验学习笔记

这份笔记梳理 `accelerate` 分支对 RoboTwin 默认训练所做的四轮加速，以及做过但没有采用的候选。读完应能回答三个问题：最初慢在哪里，各项修改为什么有效或无效，如何区分算子微测、设备时间线和完整训练步的收益。当前运行入口是 [train_robotwin_npu.sh](../train_robotwin_npu.sh)，操作与续训说明见 [NPU 训练指南](NPU_TRAINING_GUIDE.md)。

实验环境为单机 8 × Ascend 910B1、每卡 micro batch 12、global batch 96、FSDP2 分片 8、专家并行 EP=1，主要软件版本为 PyTorch/torch_npu 2.6.0、CANN 8.2.RC1。以下速度数字均来自该配置的短跑。`秒/步` 越小越好；吞吐量 = `96 ÷ 秒/步`，因此“步时减少 10%”并不等于“吞吐增加 10%”。各轮有独立基线和不同测量窗口，**不能把百分比直接相加**，也不能据此推算完整 epoch 或机器人任务成功率。

## 一、先定位瓶颈，再选择方法

最初的 [8 卡 profile](../reports/20261016_npu_profile/analysis.txt) 中，无采集窗口约 **7.11 秒/步**；带 profiler 的四步窗口约 **9.26 秒/步**，因此后者只能用来找热点。采集窗口内反向占 **56.4%**，MoE 专家范围被调用 72 次：36 层正常前向加 36 层激活重计算，合计 **2.48 秒/步**，属于前向和反向的内部范围。设备上有大量几微秒的小 MatMul；数据等待约 **0.0009 秒/步**。这指向“专家循环与任务下发”而不是数据加载。

第二轮完成后重新做的 [全 8 卡 profile](../reports/20261016_round3_profile/analysis.txt) 已是另一幅图景：采集窗口的设备时间约 **42.1% 计算、29.0% 未重叠通信、28.9% Free**；每卡每步约 **4.26 万个设备任务**，约 88% 的非通信任务短于 50 微秒。数据等待约 0.00078 秒，MoE grouped GEMM 与其权重梯度内核总和约 14 毫秒/步。这里的 14 毫秒不包含完整 MoE 路由和框架成本，但表明已不能继续把专家 GEMM 当作最大单一设备热点。`Free` 只表示时间线没有设备任务，不能直接等同 CPU 同步；通信范围也包括到达不齐和等待，不能直接等同链路传输时间。当前 EP=1，没有专家 all-to-all；跨卡通信主要来自 FSDP 参数和梯度同步。

| 轮次 | 对照窗口与同轮基线 | 最终候选 | 端到端结果 | 主要证据 |
| --- | --- | --- | --- | --- |
| 1：MoE | 第 13–29 步；原实现 7.2353 秒/步 | 原生 grouped GEMM + 显式反向 3.4706 秒/步 | 吞吐 **2.0847 倍**，步时降低 52.03% | [MoE 实验](../reports/20261016_npu_moe_optimization/analysis.txt) |
| 2：注意力、优化器、重计算 | 第 13–60 步；较快的 GMM 基线复测 3.5417 秒/步 | 四项组合 2.6042 秒/步 | 吞吐 **+36.0%**，步时降低 26.47% | [七组对照](../reports/20261016_speed_round2/final_summary.json) |
| 3：小任务与主机元数据 | 第 31–110 步；第二轮配置 2.6500 秒/步 | Muon、RMSNorm、元数据及 teacher 优化 2.4625 秒/步 | 吞吐 **+7.61%**，步时降低 7.08% | [第三轮分析](../reports/20261016_speed_round3/analysis.txt) |
| 4：通信与任务下发 | 第 31–110 步；Level 1 队列 2.4625 秒/步 | Level 2 队列 2.1750 秒/步 | 吞吐 **+13.22%**，步时降低 11.68% | [第四轮分析](../reports/20261016_comm_round4/analysis.txt) |

表中的速度使用不带 profiler 的壁钟步间隔。第 1 轮是 40 步短跑，第 2 轮每组 60 步，第 3/4 轮使用 120 步左右的独立短跑；第 3/4 轮的计时窗口都是 80 次更新、日志时间戳精度 1 秒，约有 ±0.0125 秒/步的量化不确定度。跨轮基线自然波动明显：第二轮两次原代码基线相差约 **2.9%**，所以 1% 左右的单项差异不可当作确定加速。各轮均没有用完整 epoch 的收敛或闭环成功率证明质量等价。

## 二、第一轮：把 MoE 专家循环改成合批矩阵乘

原路径按专家逐个执行小矩阵乘、Python 循环、权重拆分/拼接和反向。单层 32 专家、top-4、612 tokens、D=768、H=512 的 [隔离微基准](../reports/20261016_npu_profile/analysis.txt) 如下：

| 单层方案 | 前向 + 反向 | 结论 |
| --- | ---: | --- |
| 原实现 | 33.486 毫秒 | 基线 |
| 只提前 `unbind` 权重 | 29.180 毫秒 | 少量改善；仍有专家级小任务 |
| `unbind` 后合并输出 `cat` | 22.308 毫秒 | 较快，但没有消除主要碎片化 |
| NPU 原生 grouped GEMM + 显式反向 | 2.894 毫秒 | 此形状约 **11.6 倍**；只是单层微测 |

最终在 [npu_moe.py](../lingbotvla/ops/npu_moe.py) 使用 `torch_npu.npu_grouped_matmul` 合并专家计算，并显式写 autograd backward：输入梯度仍走 grouped GEMM，权重梯度用 `npu_grouped_matmul_add_` 做 FP32 累加，再恢复计算 dtype。这样做是因为本机 torch_npu 2.6 的原生 grouped matmul **直接调用没有训练所需的自动梯度**。专家计数与 `cumsum` 留在 NPU，路由专家 ID 用可精确表示的 FP32 值排序，避开整数排序的 AICPU 回退。权重仍是原来的 stacked 形状和名称，路由器仍用 FP32，检查点布局不变。该后端可通过 `LINGBOT_NPU_MOE_IMPL=torch` 回退。

完整 8 卡 A/B 得到 **7.2353 → 3.4706 秒/步**，峰值 rank 0 已分配显存均为 **53.31 GiB**。10 项 NPU 用例覆盖空专家、随机路由、top-1/top-4、梯度与重计算；8 卡 FSDP 梯度对照也通过。适用范围是本机 BF16 专家、非 EP、一阶反向；不要把 11.6 倍单层微测写成整训收益。原实现对照跑完 40 步后，外层 shell 因运行期间脚本被修改而报 `exit_code=2`；计时使用的是此前完整记录的训练步骤，[审计说明](../reports/20261016_npu_moe_optimization/analysis.txt) 保留了这处异常。

## 三、第二轮：利用显存余量减少重算和优化器下发

第二轮所有方案都以已经启用 GMM 的配置为起点。下表是 [七组 60 步对照](../reports/20261016_speed_round2/final_summary.json) 的第 13–60 步、每卡 12 样本结果；显存是 **rank 0 PyTorch allocator 的 peak allocated**，不是 8 卡总显存或 `npu-smi` 容量占用。

| 方案 | 秒/步 | 相对较快基线的步时变化 | rank 0 峰值 |
| --- | ---: | ---: | ---: |
| GMM 基线第一次 | 3.6458 | +2.94% | 53.31 GiB |
| 同配置基线复测 | 3.5417 | 基线 | 53.31 GiB |
| 仅监控计数合批 | 3.4792 | -1.76% | 53.31 GiB |
| 仅 FP32 SDPA | 3.5000 | -1.18% | 46.68 GiB |
| 仅 AdamW foreach | 3.1042 | -12.35% | 53.31 GiB |
| 三项组合、保留重计算 | 3.0625 | -13.53% | 46.68 GiB |
| 三项组合、关闭动作层重计算 | **2.6042** | **-26.47%** | **50.61 GiB** |

**监控计数合批。**原实现每层通过 NPU 标量判断专家 token 计数是否为零，可能形成重复的主机读取；[模型代码](../lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py) 把计数集中传回 CPU 再生成监控数据，开关为 `LINGBOT_MOE_MONITOR_BATCH_COUNTS=1`。单项时间差只有 1.76%，小于两次基线相差的 2.86%，故不能单独断言它稳定节省了 1.76%。监控口径还会随关闭重计算而变化，不能把计数减半理解成少训练样本。

**FP32 SDPA。**[joint_attention.py](../lingbotvla/ops/joint_attention.py) 用 SDPA 处理联合注意力，并直接使用 GQA 的 32 个 query head 与 8 个 KV head，避免显式扩展 KV 与写出大的 FP32 分数表。实际 B=12、序列长 337、head_dim=128 时，单张分数表约 **166 MiB**。必须保留 mask `True=可见`、全遮罩行的 V 均值及其梯度行为；不是简单替换一行 API。单层前反向微测 **8.046 → 4.316 毫秒**，但整步单项加速 1.18% 接近波动；更可靠的收益是 rank 0 峰值 **53.31 → 46.68 GiB**。[注意力候选说明](../reports/20261016_speed_round2/attention_candidate_notes.txt) 和 [硬件误差审计](../reports/20261016_speed_round2/attention_error_rule_audit.txt) 记录了边界检查。FP32 指输入接口/包装代码保持 FP32，不等于已逐条证明内核内部数学指令精度。

**AdamW foreach。**原 AdamW 对约 376 个参数逐个执行 DTensor 更新，主机调度比附着的设备计算更显著。[optimizer.py](../lingbotvla/optim/optimizer.py) 对同类参数用 foreach 批量下发，Muon 算法、参数分组和状态布局不变。第二轮单项 **3.5417 → 3.1042 秒/步**，这是比基线漂移明显更大的收益。普通 Tensor 与 DTensor 混在同一个 group 时 foreach 在本机失败，因此实现保留混合组回退；加载优化器状态后也重设运行时 foreach 开关，避免旧 checkpoint 覆盖当前选择。[foreach 设计与验证](../reports/20261016_speed_round2/adamw_foreach_notes.txt) 详述这些边界。

**关闭动作层激活重计算。**FP32 SDPA 释放的显存允许把 `ENABLE_GRAD_CKPT` 从启用改为 `false`，减少 36 个动作 decoder 层在反向时的额外前向。相对三项组合，步时 **3.0625 → 2.6042 秒**，降低 **14.97%**；rank 0 峰值从 **46.68 → 50.61 GiB**。这只改变激活保存/重算，**不会关闭训练进度 checkpoint**。默认入口又完成了 [160 步集成复测](../reports/20261016_round2_default/analysis.txt)，约 2.608 秒/步。每卡 batch 16 在现有配置曾 OOM，不能从 batch 12 的余量推断更长序列或解冻 VLM 仍可关闭重计算。

**未采用的精度候选。**BF16 注意力在放大 logit 的压力用例中，Q/K 梯度相对 L2 误差约 **3.64%/3.95%**，没有为通过而放宽门槛，生产默认仍为 `sdpa_fp32`。该失败只否定当时的候选和验证范围，不代表所有 BF16 注意力实现都不可行。

## 四、第三轮：减少 Muon 和 teacher 的细碎任务

第二轮之后的 profile 显示，主机范围内 Muon 约 **0.328 秒/步**，其中 Newton–Schulz 约 **0.146 秒**；teacher 约 **0.549 秒**，其中 SciPy 几何约 **0.129 秒**。这些是包含异步提交、同步及嵌套的范围，不能相加或直接当作可消除的壁钟时间。第三轮据此做了四项协同修改；**没有逐项完整模型消融**，所以不能把 7.61% 的总吞吐增益分配给其中某一项。[第三轮分析](../reports/20261016_speed_round3/analysis.txt) 给出了对照和验证。

| 方法与位置 | 为什么可能加速 | 验证和结果 |
| --- | --- | --- |
| [Muon 本地专家合批](../lingbotvla/optim/muon.py)，`LINGBOT_MUON_LOCAL_BATCH=8` | 相同形状的本地 3D 专家矩阵一起做 Newton–Schulz，减少逐参数下发；每个矩阵仍独立归一化，逐参数 momentum 不变 | 输入批上限 32 MiB；实际专家形状结果逐位相同；主机 Muon 范围约 328 → 224 毫秒、NS 范围约 146 → 55 毫秒 |
| [FP32 融合 RMSNorm](../lingbotvla/ops/rms_norm.py)，`LINGBOT_NPU_FUSED_RMSNORM=1` | 在原模块内部调用 `torch_npu.npu_rms_norm`，减少归一化小核 | 保留 AdaNorm 类型、FiLM、参数键与转换流程；不支持的 dtype/device 走原路径；profiler 中 RmsNorm/RmsNormGrad 各 0 → 73 次/步 |
| [CPU 整数元数据](../lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py)，`LINGBOT_HOST_METADATA=1` | 网格、位置等整数控制信息在 CPU 侧索引，避免循环读取 NPU 标量 | 修正视觉网格缓存失效条件；可学习位置嵌入仍逐步计算并保留梯度 |
| [video teacher 常量缓存](../lingbotvla/models/vla/vision_models/module_utils.py) | 复用 RGB 归一化常量和由形状决定的整数边界 | 缓存有界；不缓存图像、teacher 输出或可学习张量；profiler 中 video teacher 主机范围约 212 → 73 毫秒 |

两组全卡 profiler 的设备任务数约 **42,602 → 36,747 个/卡/步**，`BatchMatMulV2` 调用约 **2,115 → 840 次/步**。无 profiler 的整步对照为 **2.6500 → 2.4625 秒**；rank 0 峰值约 **50.61 → 50.48 GiB**。初版 RMSNorm 候选因替换模块类型破坏 AdaNorm 权重转换，在加载阶段就失败，之后改成“保留模块，只融合内部计算”。

同轮还实现并测试了按联合 decoder 两阶段调用顺序的 **FSDP 显式前向预取**。开启为 **2.4500 秒/步**，关闭为 **2.4625 秒/步**，仅约 **0.51%** 差异，低于已观察到的短跑波动，因此保留 `LINGBOT_FSDP_FORWARD_PREFETCH=0` 为默认。第三轮 profiler 中未重叠通信约 **0.990 → 1.050 秒/步**，说明减少小核没有自动解决通信暴露。AI Core 抽样均值也没有显著上升；吞吐加快不要求 `npu-smi` 利用率同步升高。

## 五、第四轮：改善通信重叠，而不是盲目减少通信次数

[第四轮对照](../reports/20261016_comm_round4/analysis.txt) 在相同 8 卡 FSDP2、batch 和模型配置下，只把 `TASK_QUEUE_ENABLE` 从 torch_npu 未设置时的 Level 1 改为 **2**。第 31–110 步 **2.4625 → 2.1750 秒/步**，吞吐 **38.98 → 44.14 样本/秒**；rank 0 峰值均为 **50.48 GiB**。同一窗口的 8 卡 `npu-smi` 抽样 AI Core 均值约 **19.05% → 22.75%**，这是稀疏采样，不是理论算力利用率 MFU。全 8 卡第 121–124 步 profiler 中，每卡每步设备任务数仍约 **36,747**，collective 次数仍是 **174 次 all-gather、74 次 reduce-scatter、6 次 all-reduce**；未重叠通信 **1.050 → 0.823 秒/步**，Free **0.718 → 0.606 秒/步**。这些带采集的秒数只用于归因，不能与无采集步时相加。证据支持任务下发与通信重叠改善；HCCL 内部具体调度机制没有单独隔离。`TASK_QUEUE_ENABLE=1` 可回退旧默认，`0` 是关队列，含义不同，见 [torch_npu 官方变量说明](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/TASK_QUEUE_ENABLE.md)。

三种通信候选没有进入默认配置：

| 候选 | 实测结果 | 决定 |
| --- | --- | --- |
| 8 卡改为 2×4 HSDP | toy 数值通过；真实 132 步 **8.775 秒/步**，all-reduce **6 → 80 次/步**，未重叠通信约 **1.05 → 6.31 秒/步**，峰值 **50.48 → 54.35 GiB** | 明显变慢，否决 |
| 合并相邻动作 decoder 层为较大的 FSDP2 单元 | toy 梯度通过；20 步 pilot 第 2–20 步吞吐 **+6.89%**，但每个 rank 都出现“某 grouped module 在 backward 前没有 forward”警告；真实模型分阶段调用这些层 | 正确性风险未消除，不进入生产 |
| Muon `all_gather_into_tensor` 替换 list all-gather | 8 卡微测逐位正确；小形状更慢，大形状更快 | 没有端到端收益证据，保持原实现 |

这里的教训是：减少 FSDP 分片范围可能增加 replica 之间的 all-reduce；减少 collective **数量**或通过 toy 测试，也不保证真实模型更快、更安全。通信 `duration sum`、未重叠通信和端到端步时是三种不同口径。

## 六、调研但尚未实施的方向

| 方向 | 当时的判断 | 状态 |
| --- | --- | --- |
| Triton-Ascend / 第三方 Triton Muon | 当前 CANN 8.2 环境还需做安装兼容、实际形状、数值和 FSDP 状态验证；不能直接套用外部内核 | 只调研，未安装，未测速度；见 [第三轮分析](../reports/20261016_speed_round3/analysis.txt) |
| `NpuFusedAdamW` 存储合并 | 本机路径要额外处理 DTensor；现有 foreach 已有端到端收益 | 未集成，未做完整 A/B |
| teacher 几何批处理、选择性重计算、编译、更大 batch | 需要分别确认语义、显存与实际时间线；batch 16 已在当前条件 OOM | 后续候选，不能写成已取得的收益 |

## 七、当前默认开关与可复核的学习路径

主入口默认使用 `LINGBOT_NPU_MOE_IMPL=gmm`、`LINGBOT_JOINT_ATTN_BACKEND=sdpa_fp32`、`LINGBOT_MOE_MONITOR_BATCH_COUNTS=1`、`LINGBOT_ADAMW_FOREACH_NPU=1`、`LINGBOT_MUON_LOCAL_BATCH=8`、`LINGBOT_HOST_METADATA=1`、`LINGBOT_NPU_FUSED_RMSNORM=1`、`LINGBOT_FSDP_FORWARD_PREFETCH=0` 和 `TASK_QUEUE_ENABLE=2`。在默认 SDPA 条件下，如未显式覆盖，脚本还会设置 `ENABLE_GRAD_CKPT=false`。直接运行 YAML 的默认行为可能不同；以实际 `lingbotvla_cli.yaml`、`train.log` 的 `[NPU speed]` 行及启动脚本为准。

复核一个新想法时，先读 [profiler 入口](../tools/profile_robotwin_npu.sh) 与相应轮次的 `analysis.txt`，看其开销是否仍在当前关键路径；再做模块输出、梯度、优化器状态检查；最后固定预训练权重、数据清单、随机种子、global batch、学习率计划，串行运行 8 卡 A/B，用**无 profiler**的相同步数窗口比较样本/秒，并在候选后重跑基线。需要区分 `npu-smi` 的 AI Core 抽样、PyTorch allocated 显存、HCCL kernel 时长和真实端到端吞吐。第四轮后训练仍有计算、通信暴露与空闲间隙；AI Core 数值低不等于没有提速空间，也不能通过增加无效计算来追求利用率。

源码与原始实验入口集中在 [训练与提速报告索引](../reports/README.md)。以上验证支持当前机器与短跑训练速度；真实 DCP 保存—恢复、完整训练收敛和 RoboTwin 闭环任务效果仍需各自验证。
