# 昇腾 NPU 训练与提速指南

这份文档帮助接手者运行当前分支、理解已经改了什么，并继续做可复核的提速实验。当前主入口是 [train_robotwin_npu.sh](../train_robotwin_npu.sh)；[train_robotwin_npu_optimized.sh](../train_robotwin_npu_optimized.sh) 只是转交给主入口的兼容脚本。

**当前证据：**在本机 8 × 昇腾 910B1、每卡 12 个样本的配置下，第二轮优化让每步时间从较快原代码复测的 **3.54 秒降到 2.60 秒**，吞吐约增加 **36%**。7 组 60 步对照和后续 160 步主入口复测已完成。它们支持当前配置的速度收益，不代表已经完成长期收敛、机器人任务成功率或新方案完整存档恢复验证。

## 1. 先准备环境、模型和数据

实测环境与 45 个相关包版本记录于 [npu_environment.json](npu_environment.json)，第二轮实验当时的精简记录见 [environment_metadata.json](../reports/20261016_speed_round2/environment_metadata.json)：

| 项目 | 本次环境 |
| --- | --- |
| 设备 | 单机 8 × Ascend 910B1，64 GiB HBM/卡 |
| CPU / Python | aarch64 / Python 3.11.10 |
| PyTorch / torch_npu | 2.6.0 / 2.6.0 |
| CANN | 8.2.RC1 |
| Transformers / NumPy | 4.57.3 / 1.26.4 |
| PyAV | 18.1.0 |

这些是已测试的组合，尚不是覆盖所有依赖的安装锁文件。先在目标机器准备兼容的 NPU 驱动、CANN、PyTorch 与 torch_npu 环境，并加载该机器的 CANN 环境。上游 README 的安装路线使用 Python 3.12、PyTorch 2.8.0 与 CUDA；[requirements.txt](../requirements.txt) 包含相应依赖，[setup.py](../setup.py) 又会读取这份清单。因此不要在已准备好的 NPU 2.6 环境直接运行 `pip install -r requirements.txt`、普通的 `pip install -e .` 或 `tools/create_train_env.sh`，否则可能替换这套环境。已有匹配环境和其他依赖后，用 `python -m pip install -e . --no-deps` 注册本仓库。

当前启动脚本按 `$HOME/anaconda3/envs/$CONDA_ENV/bin/python` 查找环境，并加载 `$HOME/anaconda3/etc/profile.d/conda.sh`。默认环境名为 `PyTorch-2.6.0`；其他目录布局需要按机器情况调整启动器。仓库通过 [NPU 兼容入口](../lingbotvla/__init__.py) 激活 `transfer_to_npu`，因此代码中出现 `torch.cuda` 不足以判断实际运行设备。

模型、数据和训练存档需要单独准备，不包含在源码交付中。参考上游 [模型下载说明](../README.md#model-download)、[数据准备](../experiment/robotwin/README.md) 和 [完整配置说明](../configs/vla/Training_Config.md)。外部用户应复制 [可移植配置模板 robotwin_npu.example.yaml](../configs/vla/robotwin/robotwin_npu.example.yaml) 为自己的 YAML，替换其中所有 `/path/to` 占位路径；原 [robotwin_npu.yaml](../configs/vla/robotwin/robotwin_npu.yaml) 保留了已验证实验机器的默认值。重点核对：

| 配置项 | 需要指向自己的资源 |
| --- | --- |
| `model.model_path` | LingBot-VLA v2 预训练模型目录 |
| `model.tokenizer_path` | Qwen3-VL-4B-Instruct 配置与 tokenizer 资源目录 |
| `train.align_params.depth.moge_path` / `morgbd_path` | MoGe / LingBot-Depth teacher 权重 |
| `train.align_params.video.ckpt_path` / `config_path` | DINO-Video teacher 权重与配置 |
| `DATA_LIST` 或 `data.train_path` | 数据清单；本例一行格式为 `robotwin /自己的数据集目录` |
| `data.robot_config_root` 与归一化统计 | 与实际机器人关节、相机、数据分布匹配 |
| `OUTPUT_DIR` / `LOG_DIR` | 检查点大盘目录 / 小日志目录 |

原配置带有实验机器路径和 OBS 目的地，它们不是可直接复用的公共服务。使用本地数据时显式设置 `OBS_DATASET_URI='' OBS_OUTPUT_DIR=''`；使用 OBS 时改成自己的桶与前缀，并在运行环境配置访问权限，不把密钥写进源码、文档或提交的配置。启动器启用了离线模型模式，训练前应把所需权重与配置准备完整。

模型路径必须修改自定义 YAML，或追加真实存在的 CLI 字段，例如 `--model.model_path /path/to/model --model.tokenizer_path /path/to/qwen`；启动器没有通用的 `MODEL_PATH` 环境变量映射。保留 `model.moe_implementation: fused`，它关系到预训练权重布局；用 `LINGBOT_NPU_MOE_IMPL=torch` 回退计算后端，与把模型权重布局改为 eager 是不同操作。

## 2. 当前默认值是什么

启动脚本的环境变量与追加 CLI 参数会覆盖 YAML。最终生效配置写入该次运行的 `lingbotvla_cli.yaml`；性能环境开关另记在 `train.log` 的 `[NPU speed]` 行。先运行 `DRY_RUN=1` 检查命令，它不会启动训练、下载或创建输出目录，但仍要求配置文件与指定环境的 Python 存在。

| 设置 | 主入口默认值 | 通俗解释 |
| --- | --- | --- |
| `LINGBOT_NPU_MOE_IMPL` | `gmm` | 把许多专家的小矩阵计算合批提交 |
| `LINGBOT_JOINT_ATTN_BACKEND` | `sdpa_fp32` | 联合注意力保持 FP32 接口，减少中间大表的生成与搬运 |
| `LINGBOT_MOE_MONITOR_BATCH_COUNTS` | `1` | 集中取回各层监控计数，减少 CPU 等待 |
| `LINGBOT_ADAMW_FOREACH_NPU` | `1` | 现有 AdamW 分支批量更新参数；不更换 Muon 算法 |
| `ENABLE_GRAD_CKPT` | 默认 SDPA 组合下为 `false` | 保留更多中间结果，减少动作层反向时的重复计算 |
| `MICRO_BATCH` / `GRAD_ACCUM` / 卡数 | `12` / `1` / `8` | 每次更新共使用 96 个样本 |
| `TRAIN_EXPERT_ONLY` | `true` | 冻结 VLM，训练动作模块及相关 query、投影、head |
| `NUM_TRAIN_EPOCHS` | `1` | 主入口默认仍是一轮；五轮需显式设置 |
| `LR` / `LR_MIN` / `LR_WARMUP_RATIO` | `5e-5` / `1e-5` / `0.02` | 预热后按 cosine 计划降低学习率 |
| `SAVE_EPOCHS` / `SAVE_STEPS` / `ENABLE_RESUME` | `1` / `0` / `true` | 每轮保存；默认不按步保存；启用续训查找 |

直接运行 YAML 与运行主脚本的默认行为不同：YAML 仍保留 `attention_implementation: eager` 和重计算开启；主脚本的运行时覆盖才构成当前优化方案。若改成 eager 注意力且未指定重计算开关，脚本会重新遵循 YAML 的重计算设置。

## 3. 新开一个五轮实验

下面使用自己的本地数据和新输出目录，避免依赖原实验的 OBS 配置。先复制 `configs/vla/robotwin/robotwin_npu.example.yaml` 到自己的配置目录，修改模型、teacher、数据与输出路径，再把 `CONFIG` 指向它。`/path/to/...` 都是需要替换的占位路径。即使模板中的 OBS 项已留空，也要显式设置下列 OBS 环境变量，因为启动脚本会覆盖 YAML 中的对应字段。

```bash
# 在仓库根目录执行；配置路径改为自己已准备好的 YAML。
export REPO_ROOT="$PWD"
export CONDA_ENV=PyTorch-2.6.0
export CONFIG=/path/to/your_robotwin_npu.yaml
export DATA_LIST=/path/to/train_robotwin.txt
export OBS_DATASET_URI=''
export OBS_OUTPUT_DIR=''

export RUN_NAME=robotwin_5ep_myexperiment_v1
export OUTPUT_DIR=/path/to/large_disk/robotwin_5ep_myexperiment_v1
export LOG_DIR="$REPO_ROOT/logs/$RUN_NAME"
export NUM_TRAIN_EPOCHS=5
export ENABLE_RESUME=false
unset MAX_STEPS STOP_AFTER_STEPS LOAD_CHECKPOINT_PATH

DRY_RUN=1 bash train_robotwin_npu.sh
# 核对打印的路径、5 epoch、batch、学习率和四项性能设置后启动：
bash train_robotwin_npu.sh
```

正式启动会占用 8 张卡；不要和另一个训练任务并行做速度对照。该启动器仍要求输出文件系统有至少默认 70 GiB 可用空间，单份本配置检查点约 60 GiB。关闭 OBS 后，本地检查点不会因为 OBS 上传验证而被清理，多轮训练需为累积文件留空间。

每个新实验使用新的 `RUN_NAME`。在未显式设置目录时，它会一起改变本地输出、日志和默认 OBS 前缀；若自己已设置 `OUTPUT_DIR`、`LOG_DIR` 或 `OBS_OUTPUT_DIR`，这些显式值优先，需要同步更新。运行名只能使用字母、数字、下划线、点和连字符，且以字母或数字开头。

### 一轮和五轮的学习率计划不同

本次数据与 global batch 96 对应每轮 5,717 次更新。一轮总计划是 5,717 步，前 114 步学习率预热；新五轮计划是 28,585 步，前 571 步预热。换数据或 batch 后必须查看日志的 `Schedule:` 行，不能照抄这些总步数。

“预热”在这里是逐渐提高学习率，让更新幅度从小到大；它与前几步编译、首次算子执行变慢是两回事。`MAX_STEPS` 会改变总学习率计划；`STOP_AFTER_STEPS` 只设置提前结束位置，保留原计划。做短跑对照时使用后者。

不要把已训练一轮的旧存档直接当作原地延长到五轮。恢复路径会检查存档里的总调度长度与训练签名，不匹配时拒绝恢复。改变总计划应新开实验，或单独设计并验证明确的迁移策略。此次五轮任务确实使用新运行名，从原预训练权重开始，见 [五轮配置审计](../reports/launch_robotwin_5ep_optimized_20261016/independent_config_audit.json)。

## 4. 保存进度与重计算，是两个不同的 checkpoint

训练可以看成四件事：前向“做题”，loss“评分”，反向计算“该往哪个方向改”，optimizer 根据梯度和学习率“真正改参数”。多卡之间还需要同步必要的信息。

**激活重计算**是在前向时少留计算草稿，反向需要时再算一次，节省显存。`ENABLE_GRAD_CKPT=false` 保留更多草稿、减少重复计算。本次只训练动作专家的配置中，这个开关实际影响动作模块的 36 层；冻结 VLM 与独立 teacher 的情况不同。它不会关闭训练进度存档。

**训练存档**保存模型、优化器状态、调度器、已完成步数和随机状态等，供以后恢复。当前 `ckpt_manager: dcp` 使用分布式检查点。带 OBS 的保存流程先保存本地，再导出与上传、校验，确认后才按设置清理本地文件。仅有模型导出权重不等同于有完整续训状态。实现见 [trainer](../tasks/vla/train_lingbotvla.py)、[训练存档辅助逻辑](../lingbotvla/utils/training_io.py) 和 [存档边界测试](../tests/test_training_obs_flow.py)。

恢复同一个实验时保留原配置、运行目录和总计划，启用 `ENABLE_RESUME=true`；需要指定本地存档时通过 `LOAD_CHECKPOINT_PATH` 传入完整存档路径。切换 AdamW foreach 执行策略时，加载后的处理会恢复本次开关，保留学习率、步数和动量状态；组件测试覆盖这一点，但不等同于真实大模型 DCP 端到端恢复已经验过。

**当前限制：**第二轮 60 步和 160 步性能验证都关闭了真实保存/恢复。后续五轮任务最后记录到第 **147 步**，随后按用户要求停止，未到第一轮保存点，也没有生成训练检查点。不能宣称这次完成了五轮训练或新方案的完整 DCP 保存—加载验证；硬中断也不保证自动留下最新进度。见 [147 步停止摘要](../experiments/npu_20261016/stopped_5ep_147/summary.json) 与 [user_stop.json](../reports/launch_robotwin_5ep_optimized_20261016/user_stop.json)。

## 5. 已测到多少收益

第二轮 7 组有效实验各运行 60 步，主指标取第 13–60 步，共 48 次更新。用第 60 步时间戳减第 12 步时间戳，再除以 48；模型加载、首次编译、前 12 步、保存与上传另计。样本吞吐为 `96 ÷ 每步秒数`。完整数值见 [结果汇总](../reports/20261016_speed_round2/final_summary.json) 与 [最终独立审计](../reports/20261016_speed_round2/final_experiment_audit.json)。

可从 [实验归档索引](../experiments/npu_20261016/README.md) 找到随仓库保留的日志、有效配置和小型原始证据。历史 JSON 中的机器绝对路径只是当时的出处记录；在其他机器复核时应以归档的相对路径索引为入口。

| 实验 | 秒/步 | 有效样本/秒 | rank 0 峰值 allocated 显存 |
| --- | ---: | ---: | ---: |
| 本轮已有 GMM 的初次基准 | 3.646 | 26.33 | 53.31 GiB |
| 同一原代码复测 | 3.542 | 27.11 | 53.31 GiB |
| 只改监控查询 | 3.479 | 27.59 | 53.31 GiB |
| 只改联合注意力 | 3.500 | 27.43 | 46.68 GiB |
| 只改原 AdamW 的 foreach | 3.104 | 30.93 | 53.31 GiB |
| 三项组合，保留重计算 | 3.063 | 31.35 | 46.68 GiB |
| 同一组合，关闭动作重计算 | **2.604** | **36.86** | **50.61 GiB** |

与较快的原代码复测相比，最终方案每步减少 **26.47%** 时间，吞吐增加 **36.0%**。只比较最后两行，关闭重计算再减少 **14.97%** 时间，代价是多用约 **3.93 GiB** 的 rank 0 峰值已分配显存；仍低于基准。这个显存值不是每张卡的峰值，也不是 `npu-smi` 的总占用或缓存预留量。

原代码两次结果相差约 2.9%。监控、注意力相对较快原代码的时间改善仅约 1.8%、1.2%，接近已观察到的波动，不能据此认定稳定的单项整步收益；注意力节省显存的效果更明确。foreach 相对较快原代码仍省约 12.4%，证据更强。只有两次原代码测量、每个候选一次短跑，不能给出长期稳定速度的置信区间。见 [复测审计](../reports/20261016_speed_round2/baseline_repeat_audit.json)。

后续直接使用主入口默认值又完成 **160 步**：13–160 步约 **2.608 秒/步**；第 115–160 步已经越过一轮计划的学习率预热，仍约 **2.609 秒/步**，记录指标均有限，rank 0 峰值仍为 50.61 GiB。这是默认入口与短期稳定性复核，不能当成新增的历史 A/B 长窗口，见 [160 步完成记录](../reports/20261016_round2_default/completion.json) 和 [配置可比性审计](../reports/20261016_round2_default/config_audit.json)。

此前单独的 MoE 合批 A/B 约 7.24 → 3.47 秒/步，属于另一轮实验；本轮已经包含 GMM，不能把跨轮数字差异继续算作新的单项收益。当前没有完成长期质量、满轮数据覆盖或机器人闭环成功率验证。

## 6. 改代码时，先知道必须保留什么

| 学习入口 | 修改与正确性边界 |
| --- | --- |
| [npu_moe.py](../lingbotvla/ops/npu_moe.py) | 原生 grouped GEMM 与显式反向；要覆盖空专家、不同路由分布和梯度 |
| [joint_attention.py](../lingbotvla/ops/joint_attention.py) | 保留 FP32 QKV 接口、GQA 共享和 mask 方向；整行不可见时显式保留原实现的 value 均值及梯度行为 |
| [optimizer.py](../lingbotvla/optim/optimizer.py) | foreach 只改变既有 AdamW 执行方式；混合普通 Tensor/DTensor 组有兼容保护，Muon 算法不变 |
| [test_moe_monitor_batch.py](../tests/test_moe_monitor_batch.py) | 监控集中取数后的正常、空计数和缺失状态边界 |
| [training_profiler.py](../lingbotvla/utils/training_profiler.py) | 把前向、反向、teacher、优化器、通信等阶段放进时间线，找真正耗时的位置 |

注意力“FP32 接口”不代表已经证明每条内部硬件指令都用完整 FP32。已核对包装代码没有显式将 QKV 转成半精度，局部输出/梯度误差很小；实际内核数学模式仍有边界。BF16 注意力候选在压力用例中未通过梯度门槛，本轮没有启用。mask 是“哪些信息可看”的规则，不能为提速改掉边界行为。

关闭重计算后，现有专家计数不再重复统计反向重算中的前向，原始专家分配数可能约减半；每步仍训练 96 个样本。当前偏置更新速度为 0，平衡损失也不读取该监控计数；不应把计数变小误认为少训练了一半数据。换配置时重新检查这些前提，见 [重计算与计数语义审计](../reports/20261016_speed_round2/checkpoint_scope_semantics_audit.json)。

## 7. 怎样做下一轮 A/B

先复制配置和记录源码版本，每次只改一个因素。固定模型初始化、数据清单、随机种子、卡数、global batch、学习率计划、loss 权重及训练范围；分别保存生效 YAML、性能环境变量、日志与数据清单哈希。所有对照串行执行，使用新目录，从同一初始化开始，避免自动恢复上一轮实验。

下面演示隔离的 60 步性能短跑，沿用前面已配置的本地模型、数据与环境。`profile_only=true` 用于跳过真实存档，空 `profile_ranks` 使所有 rank 都不采集 profiler；这里只测速度，不证明保存恢复。

```bash
RUN_NAME=ab_current_v1 \
OUTPUT_DIR=/path/to/large_disk/ab_current_v1 \
LOG_DIR="$REPO_ROOT/logs/ab_current_v1" \
NUM_TRAIN_EPOCHS=1 STOP_AFTER_STEPS=60 \
ENABLE_RESUME=false SAVE_EPOCHS=0 SAVE_STEPS=0 SAVE_HF_WEIGHTS=false \
bash train_robotwin_npu.sh \
  --train.enable_profiling true --train.profile_only true --train.profile_ranks ''
```

例如测试关闭重计算的单项影响，应保持其余开关不变，再用**另一个输出/日志目录**运行 `ENABLE_GRAD_CKPT=true` 的对照。回到第二轮开始前的 GMM 基准，才同时使用下列整套回退开关：

```bash
LINGBOT_JOINT_ATTN_BACKEND=eager \
LINGBOT_MOE_MONITOR_BATCH_COUNTS=0 \
LINGBOT_ADAMW_FOREACH_NPU=0 \
ENABLE_GRAD_CKPT=true \
DRY_RUN=1 bash train_robotwin_npu.sh
```

先跑相关 CPU 数值/控制流测试，再做小规模硬件用例，检查输出、梯度和优化器状态，最后运行整模型对照。记录失败，不为了让候选通过而放宽门槛。可参考 [测试目录](../tests)、[注意力验证工具](../tools/check_joint_attention.py)、[AdamW 验证工具](../tools/check_adamw_foreach.py) 与 [续训开关验证](../tools/check_adamw_resume.py)。模块数值测试、短跑 loss 有限和长期任务质量是三个不同层次。

定位瓶颈时另开一次 [profiler 采集](../tools/profile_robotwin_npu.sh)，不要把带采样开销的窗口直接当成速度结果。测完候选后再跑一次原代码，必要时交错复测，判断变化是否只是缓存、机器状态或调度波动。

显存像仓库，计算单元像工人：仓库装得满，不说明工人一直有有效工作。**主指标是有效样本/秒**，其次看内存与稳定性。少做重复计算后，利用率可能变化不大甚至下降；不要为了追求 100% 利用率增加无用计算。短窗 `npu-smi` 读数也不是理论算力利用率 MFU。

## 8. 下一步优先做什么

1. **补齐新方案真实保存—恢复验证，再开展长跑质量评测。**在可控短任务中生成真实 DCP 存档、重启恢复并核对步数、学习率与状态；这解决的是可持续训练能力，不能由 CPU mock 测试替代。
2. **重新采集当前最终方案的时间线。**早期 MoE 已被优化，旧瓶颈排序不能直接沿用。先量化优化器、teacher、数据等待和通信在当前一步中的占比。
3. **继续检查 Muon 内部的小矩阵与逐参数调度。**尝试按兼容形状合批前，先验证数学语义、状态与 FSDP 分片行为；不要把换优化算法当作纯速度优化。
4. **根据时间线优化 teacher/数据和通信。**只有确认对应阶段占比明显，才分别尝试减少重复预处理、调整 worker/预取，或检查 FSDP/HCCL 等待与重叠。每项保留独立对照。
5. **更大的 batch、选择性重计算、编译与低精度作为独立实验。**每卡 16 曾 OOM；当前 12 的余量不能推广到更长输入或解冻 VLM。BF16、编译或新软件版本需重新验证数值、首次启动成本与长期质量。

每次提交最好带上：改动目的、代码版本与有效配置、独立对照、有效样本/秒、显存指标的具体含义、数值检查及未验证事项。保留上游版权与 [LICENSE](../LICENSE)，把模型权重、原始数据、检查点、密钥和大体积 profiler trace 放在仓库之外。
