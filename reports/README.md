# 训练与提速报告索引

原始训练记录见 [logs](../logs/README.md)，可复算的实验小文件见 [experiments/npu_20261016](../experiments/npu_20261016/README.md)。以下入口保留实验结论、验证范围和失败记录，便于接手后核对。

| 内容 | 入口 |
| --- | --- |
| 完成的一轮训练：指标与图表 | [20261016_completed_analysis](20261016_completed_analysis/) |
| 初始 NPU 瓶颈与 profile 汇总 | [20261016_npu_profile](20261016_npu_profile/)、[20261016_bottleneck](20261016_bottleneck/) |
| 第一轮：MoE 原生 grouped GEMM 优化 | [报告](20261016_npu_moe_optimization/report.html) |
| 第二轮：监控、注意力、AdamW foreach、关闭激活重计算 | [最终实验审计](20261016_speed_round2/final_experiment_audit.json)、[重复基线审计](20261016_speed_round2/baseline_repeat_audit.json)、[候选补丁和单项检查](20261016_speed_round2/) |
| 默认启动入口的 160 步集成验证 | [完成记录](20261016_round2_default/completion.json)、[当前入口对照审计](20261016_speed_round2/current_entry_comparison_audit.json) |
| 5-epoch 任务的启动与停止 | [实际配置审计](launch_robotwin_5ep_optimized_20261016/independent_config_audit.json)、[用户停止记录](launch_robotwin_5ep_optimized_20261016/user_stop.json) |
| 本次仓库整理后的 CPU 回归检查 | [cpu_validation.json](20261016_repository_release/cpu_validation.json)、[最终测试日志](20261016_repository_release/cpu_tests_fixed.log) |

第二轮七组对照各运行 60 步；默认入口另运行 160 步并跨过学习率预热。正式 5-epoch 任务最后记录为 147 步，之后按用户要求停止，未保存该任务的训练存档；不能把它描述为完成了五轮训练。

本次 CPU 回归检查为 **64 passed、3 skipped**，另有监控测试单独通过 12 项；硬件数值检查沿用已有实验报告，没有为发布重新启动训练。短跑速度和数值检查不等同于完整训练收敛或机器人闭环效果验证。

历史报告中的机器绝对路径用于记录来源；公开浏览优先使用本索引和实验归档的相对链接。大体积原生 profiler trace、模型权重及数据集不随报告纳入 Git。
