# 原始训练记录索引

各目录保留训练日志、每步指标、生效配置及已有的 TensorBoard 事件文件。状态按落盘证据说明；分析结论见 [reports](../reports/README.md)，提速对照实验见 [experiments](../experiments/npu_20261016/README.md)。

| 任务 | 已记录状态 | 主要证据 |
| --- | --- | --- |
| `robotwin_1ep_fast` | 完成 1 epoch，共 5717 步，启动脚本退出码 0 | [指标](robotwin_1ep_fast/loss.jsonl)、[训练日志](robotwin_1ep_fast/train.log)、[保存记录](robotwin_1ep_fast/obs_checkpoints.jsonl)、[配置](robotwin_1ep_fast/lingbotvla_cli.yaml) |
| `robotwin_5ep_lr5e5` | 记录到第 658 步；按用户要求改为新的一轮实验，退出码 130 | [中断与替换说明](robotwin_5ep_lr5e5/intentional_restart.json)、[指标](robotwin_5ep_lr5e5/loss.jsonl)、[配置](robotwin_5ep_lr5e5/lingbotvla_cli.yaml) |
| `robotwin_5ep_optimized_20261016` | 计划 5 epochs，最后完整记录为第 147 步；按用户要求停止，无该任务的训练存档 | [指标](robotwin_5ep_optimized_20261016/loss.jsonl)、[训练日志](robotwin_5ep_optimized_20261016/train.log)、[停止及日志哈希摘要](../experiments/npu_20261016/stopped_5ep_147/summary.json) |

`loss.jsonl` 的 `step_time` 是训练程序记录的每步时间；比较整体速度时，实验报告还使用训练日志时间戳计算壁钟间隔。初始化步骤、保存和上传开销不能直接混入同一稳态对照窗口。

[视频解码基准](video_decode_bench_20261016.json)与[顺序复核](video_decode_bench_20261016_ordercheck.json)记录数据读取优化的局部测量，不代表完整训练速度。

配置中的模型、数据和 OBS 路径属于原运行环境。模型权重与数据集不包含在日志中；已有保存记录也不表示对应权重文件已包含在本仓库。
