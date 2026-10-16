#!/usr/bin/env bash
# 第二轮优化的兼容入口；train_robotwin_npu.sh 现已默认启用相同方案。
# 所有性能默认值由主入口统一维护，环境变量和 CLI 覆盖规则相同。
# 查看实际设置：DRY_RUN=1 bash train_robotwin_npu_optimized.sh
# 退回本轮实验前的 GMM 基线：
# LINGBOT_JOINT_ATTN_BACKEND=eager LINGBOT_MOE_MONITOR_BATCH_COUNTS=0 \
# LINGBOT_ADAMW_FOREACH_NPU=0 ENABLE_GRAD_CKPT=true bash train_robotwin_npu_optimized.sh
set -euo pipefail

lingbot_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$lingbot_script_dir}"
exec bash "$lingbot_script_dir/train_robotwin_npu.sh" "$@"
