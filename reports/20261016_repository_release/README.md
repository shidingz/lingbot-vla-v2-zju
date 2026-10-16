# 仓库整理与发布核验

本次整理保留了训练代码、原有训练日志、TensorBoard 事件、性能分析报告、失败实验和数值验证记录，并增加了[中文接手指南](../../docs/NPU_TRAINING_GUIDE.md)、[环境版本记录](../../docs/npu_environment.json)、可移植配置模板和[原始实验归档](../../experiments/npu_20261016/README.md)。上游提交历史与许可证保留。

模型权重、训练存档、数据集、编译缓存、访问凭据及大型原生 profiler 文件不进入 Git；本次整理没有删除本地原件，也没有重新启动训练。

## 本次验证

```bash
python -m pytest -q --tb=short tests --ignore=tests/test_npu_moe.py
```

在记录的 PyTorch 2.6.0 环境运行，最终结果为 **64 passed、3 skipped**。见 [cpu_tests_fixed.log](cpu_tests_fixed.log) 和 [cpu_validation.json](cpu_validation.json)。硬件数值及短跑性能验证沿用已有实验报告，本次没有再次执行 NPU 训练。

保留了首次检查的 [cpu_tests.log](cpu_tests.log)：当其他测试导入 NPU 适配层后，PyTorch profiler 被替换，导致监控测试的 CPU profiler 接口不兼容。修复仅作用于测试隔离，改用 TorchDispatchMode 统计实际标量读取算子，仍验证 36 次读取减少到 0 次、一次批量 CPU 传输。该测试单独运行也通过全部 12 项；没有为通过测试改变训练计算。

CPU 测试不等于完整训练质量评估，也不覆盖真实大模型的多卡保存恢复。当前性能证据、已验证范围及后续建议以[训练与提速指南](../../docs/NPU_TRAINING_GUIDE.md)为准。
