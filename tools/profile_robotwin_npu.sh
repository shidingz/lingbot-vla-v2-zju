#!/usr/bin/env bash
# Same 8x12 training workload; warm up before native CPU/NPU/HCCL capture.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_NAME="${RUN_NAME:-profile_robotwin_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
PROFILE_ROOT="${PROFILE_ROOT:-/cache/lingbot/profiling/$RUN_NAME}"
export OUTPUT_DIR="$PROFILE_ROOT/run"
export LOG_DIR="$PROFILE_ROOT/logs"
export OBS_OUTPUT_DIR="" OBS_DATASET_URI=""
export ENABLE_RESUME=false SAVE_HF_WEIGHTS=false SAVE_STEPS=0 SAVE_EPOCHS=0
export STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-20}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MICRO_BATCH="${MICRO_BATCH:-12}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export OMP_NUM_THREADS=1 LINGBOT_VIDEO_DECODE_THREADS=1 PYTHONUNBUFFERED=1
export NUM_WORKERS="${NUM_WORKERS:-8}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export MASTER_PORT="${MASTER_PORT:-62516}"
unset MAX_STEPS LOAD_CHECKPOINT_PATH
mkdir -p "$PROFILE_ROOT"
printf '%s\n' "$PROFILE_ROOT" > "$PROFILE_ROOT/profile_root.txt"
bash "${TRAIN_SCRIPT:-train_robotwin_npu.sh}" \
  --train.enable_profiling true --train.profile_only true \
  --train.profile_ranks "${PROFILE_RANKS:-all}" \
  --train.profile_start_step "${PROFILE_START_STEP:-13}" \
  --train.profile_end_step "${PROFILE_END_STEP:-17}" \
  --train.profile_trace_dir "$PROFILE_ROOT/trace" \
  --train.profile_record_shapes false \
  --train.profile_profile_memory false \
  --train.profile_with_stack false "$@"
