#!/usr/bin/env bash
# =============================================================================
# LingBot-VLA v2 — RoboTwin 训练启动脚本（Huawei ModelArts / Ascend 910B NPU 版）
#
# 由 train_robotwin.sh（单卡 RTX PRO 6000 98GB 版）改编：
#   - conda 环境: PyTorch-2.6.0 (torch 2.6.0 + torch_npu 2.6.0, ~/anaconda3/envs)
#   - 多卡: ASCEND_RT_VISIBLE_DEVICES / NPROC_PER_NODE 控制 (默认 8)
#   - 配置: configs/vla/robotwin/robotwin_npu.yaml（FP32 SDPA + 原生 GMM + sdpa teacher）
#   - 第二轮默认：AdamW foreach、MoE 监控合批、关闭动作层梯度重计算
#   - 默认 1 epoch，结束后保存 OBS；/cache 暂存一份检查点，验证后清理
#   - 小日志放仓库 logs/；默认从 OBS 最新完整检查点恢复
#
# 用法：
#   bash train_robotwin_npu.sh                        # 用默认参数启动
#   DRY_RUN=1 bash train_robotwin_npu.sh              # 只打印将要执行的命令
#   STOP_AFTER_STEPS=3000 bash train_robotwin_npu.sh   # 保持完整调度，在 3000 步保存退出
#   NUM_TRAIN_EPOCHS=10 bash train_robotwin_npu.sh
#   RUN_NAME=robotwin_lr3e5 LR=3e-5 LR_MIN=6e-6 ENABLE_RESUME=false bash train_robotwin_npu.sh
#   # 新实验请改 RUN_NAME：本地暂存、日志和 OBS 三个默认目录一起切换
#   LR=5e-5 OPTIMIZER=adamw bash train_robotwin_npu.sh
#   ENABLE_GRAD_CKPT=true bash train_robotwin_npu.sh # 保留第二轮优化，启用重计算以节省显存
#   # 回退到第一轮 GMM 基线（学习率、batch 与 Muon 算法不变）：
#   LINGBOT_JOINT_ATTN_BACKEND=eager LINGBOT_MOE_MONITOR_BATCH_COUNTS=0 \
#     LINGBOT_ADAMW_FOREACH_NPU=0 ENABLE_GRAD_CKPT=true bash train_robotwin_npu.sh
#   bash train_robotwin_npu.sh --train.enable_resume true    # 末尾追加原始 CLI 参数（最灵活）
#
# 规则（同原版）：
#   1) 下列参数可用「环境变量=值 bash train_robotwin_npu.sh」覆盖
#   2) 标「留空 = 用配置文件」的参数，不设就沿用 CONFIG(yaml) 里的值
#   3) 未列出的参数可通过末尾追加 --组.字段 值 传入（CLI 优先于 YAML）
#   4) dict 字段 train.align_params 用 ALIGN_MERGE 传顶层部分覆盖（脚本自动合并），
#      或 ALIGN_JSON 整包替换（必须含 mode/llm/depth 结构键）
#   5) 配置文件默认值在 configs/vla/robotwin/robotwin_npu.yaml；
#      每次训练的完整生效配置会存到 $OUTPUT_DIR/lingbotvla_cli.yaml
# =============================================================================
set -euo pipefail
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ═════════════════ 1. 实验规模（最常改） ═════════════════
REPO_ROOT="${REPO_ROOT:-/home/ma-user/work/lingbot-vla-v2}"
CONDA_ENV="${CONDA_ENV:-PyTorch-2.6.0}"
CONFIG="${CONFIG:-configs/vla/robotwin/robotwin_npu.yaml}"
RUN_NAME="${RUN_NAME:-robotwin_1ep_optimized}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "[错误] RUN_NAME 必须以字母或数字开头，后续仅允许字母、数字、下划线、点和连字符。" >&2
  exit 1
}
DATA_LIST="${DATA_LIST:-/cache/lingbot/robotwin_data/train_robotwin_clean.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/cache/lingbot/robotwin_train/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/$RUN_NAME}"
OBS_OUTPUT_DIR="${OBS_OUTPUT_DIR-obs://yw-2030-extern/Partner_Qin/robot_1014/$RUN_NAME}"
OBS_DATASET_URI="${OBS_DATASET_URI-obs://yw-2030-extern/Partner_Qin/robot_1014/datasets/RoboTwin_lerobot_v30/}"
OBS_CACHE_DIR="${OBS_CACHE_DIR:-/cache/lingbot/robotwin_data/RoboTwin_lerobot_v30}"
OBS_REMOVE_LOCAL_CHECKPOINT="${OBS_REMOVE_LOCAL_CHECKPOINT:-true}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"             # 留空：按 epoch；设置后也会改变学习率调度总步数
STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-}" # 只提前保存退出，不改变学习率调度总步数
SAVE_STEPS="${SAVE_STEPS:-0}"
SAVE_EPOCHS="${SAVE_EPOCHS:-1}"
MIN_CHECKPOINT_FREE_GB="${MIN_CHECKPOINT_FREE_GB:-70}" # 单份约 60 GiB + 余量，非累计五份
DRY_RUN="${DRY_RUN:-0}"

# ═════════════════ 2. 卡数与批大小 ═════════════════
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  NPROC_PER_NODE=$(echo "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi
MICRO_BATCH="${MICRO_BATCH:-12}"         # 12 已通过短跑；16 曾 OOM；同 global 的保守备选为 6×accum2
GRAD_ACCUM="${GRAD_ACCUM:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$((MICRO_BATCH * NPROC_PER_NODE * GRAD_ACCUM))}"

# ═════════════════ 3. 优化器 / 学习率（留空 = 用配置文件） ═════════════════
LR="${LR:-5e-5}"
LR_MIN="${LR_MIN:-1e-5}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.02}"
OPTIMIZER="${OPTIMIZER:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"
MUON_MOMENTUM="${MUON_MOMENTUM:-}"
MUON_NS_STEPS="${MUON_NS_STEPS:-}"

# ═════════════════ 4. 精度 / 显存 / 速度（留空 = 用配置文件） ═════════════════
ENABLE_FP32="${ENABLE_FP32:-}"
ENABLE_MIXED_PRECISION="${ENABLE_MIXED_PRECISION:-}"
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-}"
USE_COMPILE="${USE_COMPILE:-}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-}"
EMPTY_CACHE_STEPS="${EMPTY_CACHE_STEPS:-}"
# 第二轮默认组合已完成本机 8×12 的数值与短跑对照；各项可独立覆盖。
# 显式空 attention 值表示按 YAML 原路径运行，兼容自定义 attention 配置。
export LINGBOT_JOINT_ATTN_BACKEND="${LINGBOT_JOINT_ATTN_BACKEND-sdpa_fp32}"
export LINGBOT_MOE_MONITOR_BATCH_COUNTS="${LINGBOT_MOE_MONITOR_BATCH_COUNTS:-1}"
export LINGBOT_ADAMW_FOREACH_NPU="${LINGBOT_ADAMW_FOREACH_NPU:-1}"
case "$LINGBOT_JOINT_ATTN_BACKEND" in
  ""|eager|sdpa_fp32) ;;
  *) echo "[错误] LINGBOT_JOINT_ATTN_BACKEND 必须为空、eager 或 sdpa_fp32。" >&2; exit 1 ;;
esac
for speed_switch in LINGBOT_MOE_MONITOR_BATCH_COUNTS LINGBOT_ADAMW_FOREACH_NPU; do
  case "${!speed_switch}" in
    0|1) ;;
    *) echo "[错误] $speed_switch 必须为 0 或 1。" >&2; exit 1 ;;
  esac
done
# SDPA 省出的显存用于保留动作层激活，减少反向重计算；仍可用 env/CLI 覆盖。
# 改回其他 attention 路径时，未指定的重计算策略继续遵循 YAML。
if [[ -z "$ENABLE_GRAD_CKPT" && "$LINGBOT_JOINT_ATTN_BACKEND" == sdpa_fp32 ]]; then
  ENABLE_GRAD_CKPT=false
fi

# ═════════════════ 5. 训练范围 / 冻结（留空 = 用配置文件） ═════════════════
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-}"
FREEZE_VIT="${FREEZE_VIT:-}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"

# ═════════════════ 6. MoE 路由 / 正则（留空 = 用配置文件） ═════════════════
export LINGBOT_NPU_MOE_IMPL="${LINGBOT_NPU_MOE_IMPL:-gmm}" # NPU 原生 grouped GEMM + 显式反向；torch 可切回兼容实现
case "$LINGBOT_NPU_MOE_IMPL" in
  gmm|torch) ;;
  *) echo "[错误] LINGBOT_NPU_MOE_IMPL 必须为 gmm 或 torch。" >&2; exit 1 ;;
esac
TOKEN_TOP_K="${TOKEN_TOP_K:-}"
ROUTED_SCALING_FACTOR="${ROUTED_SCALING_FACTOR:-}"
SEQUENCE_WISE_LOSS_COEFF="${SEQUENCE_WISE_LOSS_COEFF:-}"
ROUTER_Z_LOSS_COEFF="${ROUTER_Z_LOSS_COEFF:-}"
BIAS_UPDATE_SPEED="${BIAS_UPDATE_SPEED:-}"
MOE_MONITOR_INTERVAL="${MOE_MONITOR_INTERVAL:-}"

# ═════════════════ 7. 蒸馏损失（留空 = 用配置文件） ═════════════════
ALIGN_MERGE="${ALIGN_MERGE:-}"
ALIGN_JSON="${ALIGN_JSON:-}"

# ═════════════════ 8. 数据侧（留空 = 用配置文件） ═════════════════
# PyAV/dav1d ignores OMP_NUM_THREADS; this directly limits each video codec.
export LINGBOT_VIDEO_DECODE_THREADS="${LINGBOT_VIDEO_DECODE_THREADS:-1}"
if [[ ! "$LINGBOT_VIDEO_DECODE_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[错误] LINGBOT_VIDEO_DECODE_THREADS 必须为正整数，禁止自动创建大量解码线程。" >&2
  exit 1
fi
NUM_WORKERS="${NUM_WORKERS:-}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"
IMG_SIZE="${IMG_SIZE:-}"
USE_FUTURE_IMAGE="${USE_FUTURE_IMAGE:-}"
IMAGE_AUGMENT="${IMAGE_AUGMENT:-}"
PROMPT_TYPE="${PROMPT_TYPE:-}"

# ═════════════════ 9. 日志 / 续训 ═════════════════
USE_WANDB="${USE_WANDB:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_NAME="${WANDB_NAME:-}"
ENABLE_RESUME="${ENABLE_RESUME:-true}"
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"
SAVE_HF_WEIGHTS="${SAVE_HF_WEIGHTS:-}"

# ═════════════════ 10. 其他（任意追加，原样透传） ═════════════════
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- 只读前置检查；DRY_RUN 不创建目录、不下载、不激活环境 ----
case "$CONFIG" in
  /*) CONFIG_PATH="$CONFIG" ;;
  *) CONFIG_PATH="$REPO_ROOT/$CONFIG" ;;
esac
[ -f "$CONFIG_PATH" ] || { echo "[错误] 找不到配置 $CONFIG_PATH" >&2; exit 1; }
PY_BIN="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"
[ -x "$PY_BIN" ] || { echo "[错误] 找不到环境 python: $PY_BIN" >&2; exit 1; }

# ---- align_params：ALIGN_MERGE 与配置合并；ALIGN_JSON 做完整性校验 ----
if [ -n "$ALIGN_MERGE" ]; then
  ALIGN_JSON=$("$PY_BIN" -c '
import json, sys, yaml
base = yaml.safe_load(open(sys.argv[1]))["train"]["align_params"]
merged = {**base, **json.loads(sys.argv[2])}
print(json.dumps(merged))
' "$CONFIG_PATH" "$ALIGN_MERGE") || { echo "[错误] ALIGN_MERGE 合并失败，请检查 JSON 格式" >&2; exit 1; }
fi
if [ -n "$ALIGN_JSON" ]; then
  "$PY_BIN" -c '
import json, sys
d = json.loads(sys.argv[1])
missing = [k for k in ("mode", "llm", "depth") if k not in d]
if missing:
    print(f"缺少结构键 {missing}；align_params 是整包替换，缺结构键会导致模型加载失败。"
          f"只改损失权重请用 ALIGN_MERGE", file=sys.stderr)
    sys.exit(1)
' "$ALIGN_JSON" || { echo "[错误] ALIGN_JSON 校验失败（需为完整 JSON 且含 mode/llm/depth）" >&2; exit 1; }
fi

# ---- 组装参数（固定项 + 可选覆盖） ----
ARGS=(
  --data.train_path "$DATA_LIST"
  --data.data_name multi
  --data.obs_dataset_uri "$OBS_DATASET_URI"
  --data.obs_cache_dir "$OBS_CACHE_DIR"
  --train.output_dir "$OUTPUT_DIR"
  --train.logging_dir "$LOG_DIR"
  --train.obs_output_dir "$OBS_OUTPUT_DIR"
  --train.obs_remove_local_checkpoint "$OBS_REMOVE_LOCAL_CHECKPOINT"
  --train.num_train_epochs "$NUM_TRAIN_EPOCHS"
  --train.save_steps "$SAVE_STEPS"
  --train.save_epochs "$SAVE_EPOCHS"
  --train.micro_batch_size "$MICRO_BATCH"
  --train.gradient_accumulation_steps "$GRAD_ACCUM"
  --train.global_batch_size "$GLOBAL_BATCH"
)
add_opt() { [ -n "$2" ] && ARGS+=(--"$1" "$2"); return 0; }
add_opt train.max_steps "$MAX_STEPS"
add_opt train.stop_after_steps "$STOP_AFTER_STEPS"
add_opt train.lr "$LR"
add_opt train.lr_min "$LR_MIN"
add_opt train.lr_decay_style "$LR_DECAY_STYLE"
add_opt train.lr_warmup_ratio "$LR_WARMUP_RATIO"
add_opt train.optimizer "$OPTIMIZER"
add_opt train.weight_decay "$WEIGHT_DECAY"
add_opt train.max_grad_norm "$MAX_GRAD_NORM"
add_opt train.muon_momentum "$MUON_MOMENTUM"
add_opt train.muon_ns_steps "$MUON_NS_STEPS"
add_opt train.enable_fp32 "$ENABLE_FP32"
add_opt train.enable_mixed_precision "$ENABLE_MIXED_PRECISION"
add_opt train.enable_gradient_checkpointing "$ENABLE_GRAD_CKPT"
add_opt train.use_compile "$USE_COMPILE"
add_opt train.enable_activation_offload "$ENABLE_ACTIVATION_OFFLOAD"
add_opt train.empty_cache_steps "$EMPTY_CACHE_STEPS"
add_opt train.freeze_vision_encoder "$FREEZE_VISION_ENCODER"
add_opt train.freeze_vit "$FREEZE_VIT"
add_opt train.train_expert_only "$TRAIN_EXPERT_ONLY"
add_opt train.token_top_k "$TOKEN_TOP_K"
add_opt train.routed_scaling_factor "$ROUTED_SCALING_FACTOR"
add_opt train.sequence_wise_loss_coeff "$SEQUENCE_WISE_LOSS_COEFF"
add_opt train.router_z_loss_coeff "$ROUTER_Z_LOSS_COEFF"
add_opt train.bias_update_speed "$BIAS_UPDATE_SPEED"
add_opt train.moe_monitor_interval "$MOE_MONITOR_INTERVAL"
add_opt train.align_params "$ALIGN_JSON"
add_opt data.num_workers "$NUM_WORKERS"
add_opt data.prefetch_factor "$PREFETCH_FACTOR"
add_opt data.img_size "$IMG_SIZE"
add_opt data.use_future_image "$USE_FUTURE_IMAGE"
add_opt data.image_augment "$IMAGE_AUGMENT"
add_opt data.prompt_type "$PROMPT_TYPE"
add_opt train.use_wandb "$USE_WANDB"
add_opt train.wandb_project "$WANDB_PROJECT"
add_opt train.wandb_name "$WANDB_NAME"
add_opt train.enable_resume "$ENABLE_RESUME"
add_opt train.load_checkpoint_path "$LOAD_CHECKPOINT_PATH"
add_opt train.save_hf_weights "$SAVE_HF_WEIGHTS"

# EXTRA_ARGS 只按 shell 引号规则拆分，不执行命令替换、变量展开或 eval。
EXTRA_ARGS_JSON=$("$PY_BIN" -c 'import json, shlex, sys; print(json.dumps(shlex.split(sys.argv[1])))' "$EXTRA_ARGS") || {
  echo "[错误] EXTRA_ARGS 引号格式无效" >&2
  exit 1
}
mapfile -d '' -t EXTRA_ARGV < <("$PY_BIN" -c 'import json, sys; sys.stdout.write("".join(s + "\0" for s in json.loads(sys.argv[1])))' "$EXTRA_ARGS_JSON")
ARGS+=("${EXTRA_ARGV[@]}" "$@")

# 以最终 CLI 覆盖值做摘要、容量检查和 tee，保证末尾追加参数真正生效。
effective_arg() {
  local key="$1" value="$2" i
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--$key" ]]; then
      if ((i + 1 >= ${#ARGS[@]})); then
        echo "[错误] --$key 缺少参数值" >&2
        return 1
      fi
      value="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" == "--$key="* ]]; then
      value="${ARGS[i]#*=}"
    fi
  done
  printf '%s\n' "$value"
}
OUTPUT_DIR=$(effective_arg train.output_dir "$OUTPUT_DIR")
LOG_DIR=$(effective_arg train.logging_dir "$LOG_DIR")
OBS_OUTPUT_DIR=$(effective_arg train.obs_output_dir "$OBS_OUTPUT_DIR")
OBS_DATASET_URI=$(effective_arg data.obs_dataset_uri "$OBS_DATASET_URI")
OBS_CACHE_DIR=$(effective_arg data.obs_cache_dir "$OBS_CACHE_DIR")
DATA_LIST=$(effective_arg data.train_path "$DATA_LIST")
NUM_TRAIN_EPOCHS=$(effective_arg train.num_train_epochs "$NUM_TRAIN_EPOCHS")
SAVE_EPOCHS=$(effective_arg train.save_epochs "$SAVE_EPOCHS")
SAVE_STEPS=$(effective_arg train.save_steps "$SAVE_STEPS")
MAX_STEPS=$(effective_arg train.max_steps "$MAX_STEPS")
STOP_AFTER_STEPS=$(effective_arg train.stop_after_steps "$STOP_AFTER_STEPS")
MICRO_BATCH=$(effective_arg train.micro_batch_size "$MICRO_BATCH")
GRAD_ACCUM=$(effective_arg train.gradient_accumulation_steps "$GRAD_ACCUM")
GLOBAL_BATCH=$(effective_arg train.global_batch_size "$GLOBAL_BATCH")
ENABLE_RESUME=$(effective_arg train.enable_resume "$ENABLE_RESUME")
ENABLE_GRAD_CKPT=$(effective_arg train.enable_gradient_checkpointing "$ENABLE_GRAD_CKPT")
LR=$(effective_arg train.lr "$LR")
LR_MIN=$(effective_arg train.lr_min "$LR_MIN")
LR_WARMUP_RATIO=$(effective_arg train.lr_warmup_ratio "$LR_WARMUP_RATIO")
MASTER_PORT="${MASTER_PORT:-62500}"
COMMAND=(torchrun --nnodes=1 --nproc-per-node "$NPROC_PER_NODE"
  --master-addr=127.0.0.1 --master-port="$MASTER_PORT"
  tasks/vla/train_lingbotvla.py "$CONFIG_PATH" "${ARGS[@]}")

# ---- 启动摘要 ----
cat <<EOF
============================================================
 实验名称    : $RUN_NAME
 配置        : $CONFIG_PATH
 OBS 数据    : ${OBS_DATASET_URI:-关闭，使用本地数据清单}
 数据缓存    : $OBS_CACHE_DIR
 数据清单    : $DATA_LIST
 检查点暂存  : $OUTPUT_DIR
 小日志      : $LOG_DIR
 OBS 结果    : ${OBS_OUTPUT_DIR:-关闭}
 卡数        : NPROC_PER_NODE=$NPROC_PER_NODE
 训练轮数    : $NUM_TRAIN_EPOCHS epoch
 保存        : 每 $SAVE_EPOCHS epoch；步间隔=$SAVE_STEPS（0 表示关闭按步保存）
 最大步数    : ${MAX_STEPS:-未设置，按完整 epoch 训练}
 提前停止步  : ${STOP_AFTER_STEPS:-未设置}（不改变学习率调度总长度）
 batch       : micro=$MICRO_BATCH×卡=$NPROC_PER_NODE  accum=$GRAD_ACCUM  global=$GLOBAL_BATCH
 NPU MoE     : $LINGBOT_NPU_MOE_IMPL
 联合注意力  : ${LINGBOT_JOINT_ATTN_BACKEND:-配置原值}
 统计合批    : $LINGBOT_MOE_MONITOR_BATCH_COUNTS
 参数合批    : $LINGBOT_ADAMW_FOREACH_NPU（现有 AdamW 分支）
 重计算      : ${ENABLE_GRAD_CKPT:-配置文件}（用于节省显存，与保存训练进度不同）
 视频解码    : PyAV 每个解码器 $LINGBOT_VIDEO_DECODE_THREADS 线程，仅转换选中的帧
 学习率      : $LR → $LR_MIN，warmup ratio=$LR_WARMUP_RATIO
 续训        : $ENABLE_RESUME（启用时检查 OBS 最新完整检查点）
 暂存盘要求  : 至少 ${MIN_CHECKPOINT_FREE_GB} GiB 可用，按单份检查点估算
============================================================
EOF

# ---- 启动 ----
if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY_RUN] 仅显示命令；未创建目录、下载数据或启动训练。"
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'export LINGBOT_NPU_MOE_IMPL=%q\n' "$LINGBOT_NPU_MOE_IMPL"
  printf 'export LINGBOT_JOINT_ATTN_BACKEND=%q\n' "$LINGBOT_JOINT_ATTN_BACKEND"
  printf 'export LINGBOT_MOE_MONITOR_BATCH_COUNTS=%q\n' "$LINGBOT_MOE_MONITOR_BATCH_COUNTS"
  printf 'export LINGBOT_ADAMW_FOREACH_NPU=%q\n' "$LINGBOT_ADAMW_FOREACH_NPU"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if pgrep -f "train_lingbot[v]la.py" >/dev/null 2>&1; then
  echo "[错误] 已有训练进程在跑，请先确认现有任务状态。" >&2
  exit 1
fi
cd "$REPO_ROOT"
if [ -z "$OBS_DATASET_URI" ]; then
  [ -f "$DATA_LIST" ] || { echo "[错误] 找不到数据清单 $DATA_LIST" >&2; exit 1; }
  grep -q "robotwin " "$DATA_LIST" || { echo "[错误] 数据清单格式应为每行: robotwin <数据集目录>" >&2; exit 1; }
fi
# 检查真正的输出文件系统；即使目录未创建，也查询其最近的现存父目录。
"$PY_BIN" - "$OUTPUT_DIR" "$MIN_CHECKPOINT_FREE_GB" <<'PY'
import pathlib
import shutil
import sys

target = pathlib.Path(sys.argv[1]).expanduser().resolve()
if sys.argv[1].startswith("obs://"):
    raise SystemExit("[错误] OUTPUT_DIR 必须是本地大盘路径；OBS 目的地请用 OBS_OUTPUT_DIR。")
minimum = float(sys.argv[2])
if minimum < 60:
    raise SystemExit("[错误] 暂存盘检查下限不能小于 60 GiB；默认要求 70 GiB。")
probe = target
while not probe.exists():
    probe = probe.parent
free_gib = shutil.disk_usage(probe).free / (1024 ** 3)
if free_gib < minimum:
    raise SystemExit(
        f"[错误] 检查点暂存盘 {probe} 仅余 {free_gib:.1f} GiB，至少需要 {minimum:g} GiB。"
        "请将 OUTPUT_DIR 放到 /cache 等大盘；OBS 上传成功后才可清理本地检查点。"
    )
print(f"暂存盘检查通过：{probe} 可用 {free_gib:.1f} GiB。")
PY

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# 环境开关不在 YAML 中；追加到正式训练日志，便于复现和续训核对。
printf '[NPU speed] moe=%s joint_attention=%s monitor_batch=%s adamw_foreach=%s gradient_checkpointing=%s\n' \
  "$LINGBOT_NPU_MOE_IMPL" "${LINGBOT_JOINT_ATTN_BACKEND:-from_config}" \
  "$LINGBOT_MOE_MONITOR_BATCH_COUNTS" "$LINGBOT_ADAMW_FOREACH_NPU" \
  "${ENABLE_GRAD_CKPT:-from_config}" | tee -a "$LOG_DIR/train.log"
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_DIR/train.log"

echo "日志已保存: $LOG_DIR/train.log"
echo "TensorBoard: tensorboard --logdir $LOG_DIR/runs --port 6006"
