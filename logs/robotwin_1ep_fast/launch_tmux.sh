#!/usr/bin/env bash
set -uo pipefail
cd /home/ma-user/work/lingbot-vla-v2
export RUN_NAME=robotwin_1ep_fast
export NPROC_PER_NODE=8 MICRO_BATCH=12 GRAD_ACCUM=1 GLOBAL_BATCH=96
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_TRAIN_EPOCHS=1 LR=5e-5 LR_MIN=1e-5 LR_WARMUP_RATIO=0.02
export SAVE_EPOCHS=1 SAVE_STEPS=0 DRY_RUN=0
export TRAIN_EXPERT_ONLY=true ENABLE_RESUME=false SAVE_HF_WEIGHTS=true
export ENABLE_FP32=true ENABLE_MIXED_PRECISION=true ENABLE_GRAD_CKPT=true USE_COMPILE=false
export NUM_WORKERS=8 PREFETCH_FACTOR=4 LINGBOT_VIDEO_DECODE_THREADS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export OUTPUT_DIR=/cache/lingbot/robotwin_train/robotwin_1ep_fast
export LOG_DIR=/home/ma-user/work/lingbot-vla-v2/logs/robotwin_1ep_fast
export OBS_OUTPUT_DIR=obs://yw-2030-extern/Partner_Qin/robot_1014/robotwin_1ep_fast
export OBS_DATASET_URI=obs://yw-2030-extern/Partner_Qin/robot_1014/datasets/RoboTwin_lerobot_v30/
export OBS_CACHE_DIR=/cache/lingbot/robotwin_data/RoboTwin_lerobot_v30
export CONFIG=configs/vla/robotwin/robotwin_npu.yaml
unset MAX_STEPS STOP_AFTER_STEPS EXTRA_ARGS ALIGN_MERGE ALIGN_JSON LOAD_CHECKPOINT_PATH
bash /home/ma-user/work/lingbot-vla-v2/train_robotwin_npu.sh
training_exit_code=$?
printf '%s\n' "$training_exit_code" > "$LOG_DIR/exit_code"
printf '\nTraining process exited with code %s\n' "$training_exit_code"
exit "$training_exit_code"
