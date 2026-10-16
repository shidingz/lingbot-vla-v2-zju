# 2026-10-16 NPU 训练与提速实验记录

这里保留可复核的小型原始记录：每步指标、训练日志、生效配置、启动清单、资源采样和退出状态。139 个源文件共 6,882,785 字节，均为原件的逐字节副本；原始本地文件没有移动或删除。

[archive_manifest.json](archive_manifest.json) 逐项记录原始绝对路径、仓库相对路径、SHA-256 和文件大小。绝对路径是历史来源信息；浏览和复算请使用仓库相对路径，不需要原来的 `/cache` 目录。

## 记录索引

| 实验 | 原始记录 | 结论与边界 |
| --- | --- | --- |
| 第一轮 MoE：逐专家实现与原生 grouped GEMM | [torch](moe_optimization_20261016/ab/torch/)、[gmm](moe_optimization_20261016/ab/gmm/)、[资源采样](moe_optimization_20261016/ab/resources.jsonl) | 各 40 步；统计未受 profiler 影响的 13–29 步。torch 在完成训练后发生 shell 收尾错误，原始退出码 2 与 [完成情况审计](moe_optimization_20261016/ab/torch/training_completion.json) 一并保留。见[分析报告](../../reports/20261016_npu_moe_optimization/report.html)。 |
| 第二轮：七组 60 步对照 | [speed_round2_20261016](speed_round2_20261016/) | `gmm_baseline_v2`、`baseline_repeat`、`monitor`、`attention`、`adamw_foreach`、`combined`、`combined_no_checkpoint` 均完成 60 步。另保留未成功开始训练的 `gmm_baseline`，分析时排除。见[最终审计](../../reports/20261016_speed_round2/final_experiment_audit.json)。 |
| 默认入口 160 步集成验证 | [default_160](round2_default_20261016_185821/default_160/) | 使用默认性能开关；跨过第 114 步学习率预热。115–160 步壁钟约 2.609 秒/步。未执行完整大小的训练存档保存或恢复。见[完成记录](../../reports/20261016_round2_default/completion.json)。 |
| 初始瓶颈采样 | [robotwin_20261016_1530](robotwin_20261016_1530/) | 保留训练/资源采样与少量诊断日志；原生大体积 trace 未纳入。见[profile 汇总](../../reports/20261016_npu_profile/profile_summary.json)。 |
| 正式 5-epoch 任务，用户要求停止 | [停止摘要](stopped_5ep_147/summary.json)、[正式训练日志](../../logs/robotwin_5ep_optimized_20261016/)、[停止记录](../../reports/launch_robotwin_5ep_optimized_20261016/user_stop.json) | 最后完整记录为第 147 步，未达到第一个 epoch 保存点；没有该任务的可恢复训练存档。摘要记录原始日志 SHA-256。 |

第二轮的稳态比较采用相同的 13–60 步窗口；壁钟间隔是第 12 步结束至第 60 步结束，共 48 次更新。较快的重复基线约 3.542 秒/步，组合关闭重计算后约 2.604 秒/步，耗时降低 26.47%。具体数值差异、重复基线波动和显存测量范围见审计文件；短跑结果不等同于完整训练质量或机器人任务成功率验证。

`combined_no_checkpoint` 指关闭**激活重计算**以提速，不是关闭训练进度保存。`stopped_5ep_147` 没有保存训练存档，则是因为任务在首个 epoch 完成前停止；这两个概念不同。

## 从归档复算

在仓库根目录运行，安装 Python 与 PyYAML 即可。这些命令只分析文件，不初始化 NPU，也不启动训练。

```bash
python tools/analyze_robotwin_moe_ab.py \
  experiments/npu_20261016/moe_optimization_20261016/ab \
  --output /tmp/lingbot_moe_ab_reanalysis

python reports/20261016_speed_round2/analyze_round2.py \
  experiments/npu_20261016/speed_round2_20261016 \
  --baseline baseline_repeat \
  --output /tmp/lingbot_round2_reanalysis.json

python reports/20261016_speed_round2/analyze_round2.py \
  experiments/npu_20261016/round2_default_20261016_185821 \
  --baseline default_160 --min-steps 160 \
  --window 13:160 --window 115:160 \
  --output /tmp/lingbot_default160_reanalysis.json
```

三个命令已用本归档验证通过。历史 manifest 内的 `source_snapshot` 指向当时机器上的源码快照；其他机器没有这些目录时，无法再次验证该快照文件，但仍可复算归档中的时间和数值记录。生产代码、历史候选补丁及单项检查结果见仓库源码和 [reports/20261016_speed_round2](../../reports/20261016_speed_round2/)。

## 没有纳入的内容

- 原生 profiler trace、驱动二进制记录、重复 tokenizer 文件、源码快照副本，以及模型和优化器权重。原 `/cache/lingbot/profiling` 约 24.58 GB，大部分为原生 trace；本归档没有把它们复制进 Git。
- 外部数据集与预训练模型。配置中的历史模型、数据和 OBS 路径只是来源信息，不是公开下载地址，也不保证其他环境直接可用。
- 全卡精确峰值显存与设备利用率统计。现有显存日志主要来自 rank 0；资源采样主要覆盖卡 0/4，采样并不同步，不能当作 MFU。

归档保留原始错误状态和诊断上下文，未修改原始数值。文本凭据模式扫描未发现命中，但这种扫描不构成对所有敏感信息的完整证明。仓库内所有导航链接使用相对路径；原生 profiler 报告如需重新生成，需要在具备数据、模型和 NPU 的环境中重新采样。
