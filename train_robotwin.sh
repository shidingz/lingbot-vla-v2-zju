#!/usr/bin/env bash
# =============================================================================
# LingBot-VLA v2 — RoboTwin 训练启动脚本（单卡 RTX PRO 6000 98GB）
#
# 用法：
#   bash train_robotwin.sh                        # 用默认参数启动
#   DRY_RUN=1 bash train_robotwin.sh              # 只打印将要执行的命令，不启动
#   MAX_STEPS=3000 SAVE_STEPS=1000 bash train_robotwin.sh
#   LR=5e-5 OPTIMIZER=adamw bash train_robotwin.sh
#   bash train_robotwin.sh --train.enable_resume true    # 末尾追加原始 CLI 参数（最灵活）
#
# 规则：
#   1) 所有参数都可以在下面用「环境变量=值 bash train_robotwin.sh」覆盖
#   2) 标「留空 = 用配置文件」的参数，不设就沿用 CONFIG(yaml) 里的值
#   3) 任何没在上面列出的参数，都可以通过末尾追加 --组.字段 值 传入（CLI 优先于 YAML）
#   4) dict 类型字段（目前只有 train.align_params）用 ALIGN_MERGE 传部分覆盖（脚本自动合并），
#      或用 ALIGN_JSON 整包替换（必须含 mode/llm/depth 结构键）
#   5) 配置文件的默认值在 configs/vla/robotwin/robotwin_local.yaml；
#      每次训练的完整生效配置会存到 $OUTPUT_DIR/lingbotvla_cli.yaml（可直接当配置文件复用）
#
# 常用组合：
#   5 任务子集 1 epoch :  MAX_STEPS=1468  DATA_LIST=/root/robotwin_data/train_robotwin_5tasks.txt
#   全量 clean 1 epoch :  MAX_STEPS=17153 DATA_LIST=/root/robotwin_data/train_robotwin_clean.txt
# =============================================================================
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ═════════════════ 1. 实验规模（最常改） ═════════════════
REPO_ROOT="${REPO_ROOT:-/root/lingbot-vla-v2}"
CONDA_ENV="${CONDA_ENV:-lingbotvla}"
CONFIG="${CONFIG:-configs/vla/robotwin/robotwin_local.yaml}"
DATA_LIST="${DATA_LIST:-/root/robotwin_data/train_robotwin_5tasks.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/robotwin_train/exp_5tasks_only_action}"
MAX_STEPS="${MAX_STEPS:-10000}"           # 5 任务子集约 1468 步/epoch；全量 clean 约 17153 步/epoch
SAVE_STEPS="${SAVE_STEPS:-2500}"         # 每个 checkpoint 约 73GB（50GB DCP + 24GB hf_ckpt）
DRY_RUN="${DRY_RUN:-0}"                  # 1 = 只打印命令

# ═════════════════ 2. 批大小（单卡实测上限 32） ═════════════════
MICRO_BATCH="${MICRO_BATCH:-32}"         # 32=实测最大（48/64 OOM）；峰值显存 ~88GB
GRAD_ACCUM="${GRAD_ACCUM:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$((MICRO_BATCH * GRAD_ACCUM))}"

# ═════════════════ 3. 优化器 / 学习率（留空 = 用配置文件） ═════════════════
LR="${LR:-5e-5}"                             # 默认 1e-4 | 官方按 global batch 1024 调的，小 batch 可试 5e-5
LR_MIN="${LR_MIN:-1e-5}"                       # 默认 5e-5
LR_DECAY_STYLE="${LR_DECAY_STYLE:-}"       # 默认 cosine | 可选 cosine / constant
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-}"     # 默认 0 | 0~1
OPTIMIZER="${OPTIMIZER:-}"                 # 默认 muon | 可选 muon / adamw / dist_muon
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"           # 默认 0
MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"         # 默认 1.0
MUON_MOMENTUM="${MUON_MOMENTUM:-}"         # 默认 0.95
MUON_NS_STEPS="${MUON_NS_STEPS:-}"         # 默认 5

# ═════════════════ 4. 精度 / 显存 / 速度（留空 = 用配置文件） ═════════════════
ENABLE_FP32="${ENABLE_FP32:-}"                     # 默认 true  | 动作专家 fp32
ENABLE_MIXED_PRECISION="${ENABLE_MIXED_PRECISION:-}" # 默认 true
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-}"           # 默认 true  | 梯度检查点（关掉更快但显存涨）
USE_COMPILE="${USE_COMPILE:-}"                     # 默认 false | torch.compile（实测本卡反而慢）
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-}" # 默认 false
EMPTY_CACHE_STEPS="${EMPTY_CACHE_STEPS:-}"         # 默认 500  | 每 N 步清显存缓存

# ═════════════════ 5. 训练范围 / 冻结（留空 = 用配置文件） ═════════════════
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-}" # 默认 false | true = 冻结视觉塔（省显存）
FREEZE_VIT="${FREEZE_VIT:-}"                       # 默认 false
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"         # 默认 false | true = 只训动作专家

# ═════════════════ 6. MoE 路由 / 正则（留空 = 用配置文件） ═════════════════
TOKEN_TOP_K="${TOKEN_TOP_K:-}"                     # 默认 4   | 每 token 激活的专家数
ROUTED_SCALING_FACTOR="${ROUTED_SCALING_FACTOR:-}" # 默认 4.0
SEQUENCE_WISE_LOSS_COEFF="${SEQUENCE_WISE_LOSS_COEFF:-}"   # 默认 1e-3 | 序列级均衡损失
ROUTER_Z_LOSS_COEFF="${ROUTER_Z_LOSS_COEFF:-}"             # 默认 1e-4 | router z-loss
BIAS_UPDATE_SPEED="${BIAS_UPDATE_SPEED:-}"                 # 默认 0    | loss-free 负载均衡
MOE_MONITOR_INTERVAL="${MOE_MONITOR_INTERVAL:-}"           # 默认 1000 | MoE 监控打印间隔

# ═════════════════ 7. 蒸馏损失（留空 = 用配置文件） ═════════════════
# ALIGN_MERGE（推荐）：只写要改的键，脚本自动与配置里的 align_params 合并
#   例：ALIGN_MERGE='{"depth_loss_weight": 0.002, "future_video_loss_weight": 0.002}'
#   默认：depth_loss_weight 0.004 / future_depth_loss_weight 0.004 / future_video_loss_weight 0.004
# ALIGN_JSON：整包替换（必须包含 mode/llm/depth 等全部结构键，否则模型加载会失败）
ALIGN_MERGE="${ALIGN_MERGE:-}"           # 默认 空（用配置文件里的完整 align_params）
ALIGN_JSON="${ALIGN_JSON:-}"             # 默认 空

# ═════════════════ 8. 数据侧（留空 = 用配置文件） ═════════════════
NUM_WORKERS="${NUM_WORKERS:-}"           # 默认 8
PREFETCH_FACTOR="${PREFETCH_FACTOR:-}"   # 默认 4
IMG_SIZE="${IMG_SIZE:-}"                 # 默认 256
USE_FUTURE_IMAGE="${USE_FUTURE_IMAGE:-}" # 默认 true（蒸馏未来帧必需，改了 align_params 会报错）
IMAGE_AUGMENT="${IMAGE_AUGMENT:-}"       # 默认 false
PROMPT_TYPE="${PROMPT_TYPE:-}"           # 默认 global | 可选 global / subtask

# ═════════════════ 9. 日志 / 续训 ═════════════════
USE_WANDB="${USE_WANDB:-}"               # 默认 false
WANDB_PROJECT="${WANDB_PROJECT:-}"       # 默认 lingbotvla
WANDB_NAME="${WANDB_NAME:-}"             # 默认 空（None）
ENABLE_RESUME="${ENABLE_RESUME:-}"       # 默认 true  | 从 OUTPUT_DIR 最新 checkpoint 续训
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"   # 默认 空 | 指定具体 checkpoint 路径续训
SAVE_HF_WEIGHTS="${SAVE_HF_WEIGHTS:-}"   # 默认 true  | 是否导出 hf_ckpt（推理用）

# ═════════════════ 10. 其他（任意追加，原样透传） ═════════════════
# 例：EXTRA_ARGS='--train.save_hf_weights true --data.max_seq_len 2048'
EXTRA_ARGS="${EXTRA_ARGS:-}"             # 默认 空

# ---- 前置检查 ----
if [ "$DRY_RUN" != "1" ] && pgrep -f "train_lingbot[v]la.py" >/dev/null 2>&1; then
  echo "[错误] 已有训练进程在跑（本机只有一块 GPU，同时跑会 OOM）。" >&2
  echo "       先停止：pkill -f 'train_lingbotvla.py'（或 kill <PID>，见 pgrep -af train_lingbotvla）" >&2
  echo "       要接着上次进度：ENABLE_RESUME=true 且用同一个 OUTPUT_DIR。" >&2
  exit 1
fi
[ -f "$REPO_ROOT/$CONFIG" ] || { echo "[错误] 找不到配置 $REPO_ROOT/$CONFIG" >&2; exit 1; }
[ -f "$DATA_LIST" ]        || { echo "[错误] 找不到数据清单 $DATA_LIST" >&2; exit 1; }
grep -q "robotwin " "$DATA_LIST" || { echo "[错误] 数据清单格式应为每行: robotwin <数据集目录>" >&2; exit 1; }

# ---- align_params：ALIGN_MERGE 与配置合并；ALIGN_JSON 做完整性校验 ----
if [ -n "$ALIGN_MERGE" ]; then
  ALIGN_JSON=$(conda run -n "$CONDA_ENV" python -c '
import json, sys, yaml
base = yaml.safe_load(open(sys.argv[1]))["train"]["align_params"]
merged = {**base, **json.loads(sys.argv[2])}
print(json.dumps(merged))
' "$REPO_ROOT/$CONFIG" "$ALIGN_MERGE") || { echo "[错误] ALIGN_MERGE 合并失败，请检查 JSON 格式" >&2; exit 1; }
fi
if [ -n "$ALIGN_JSON" ]; then
  conda run -n "$CONDA_ENV" python -c '
import json, sys
d = json.loads(sys.argv[1])
missing = [k for k in ("mode", "llm", "depth") if k not in d]
if missing:
    print(f"缺少结构键 {missing}；align_params 是整包替换，缺结构键会导致模型加载失败。"
          f"只改损失权重请用 ALIGN_MERGE (例如 ALIGN_MERGE=\x27{{\"depth_loss_weight\": 0.002}}\x27)", file=sys.stderr)
    sys.exit(1)
' "$ALIGN_JSON" || { echo "[错误] ALIGN_JSON 校验失败（需为完整 JSON 且含 mode/llm/depth）" >&2; exit 1; }
fi

# ---- 组装参数（固定项 + 可选覆盖） ----
ARGS=(
  --data.train_path "$DATA_LIST"
  --data.data_name multi
  --train.output_dir "$OUTPUT_DIR"
  --train.max_steps "$MAX_STEPS"
  --train.save_steps "$SAVE_STEPS"
  --train.micro_batch_size "$MICRO_BATCH"
  --train.gradient_accumulation_steps "$GRAD_ACCUM"
  --train.global_batch_size "$GLOBAL_BATCH"
)
add_opt() { [ -n "$2" ] && ARGS+=(--"$1" "$2"); return 0; }
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

# ---- 磁盘占用预估 ----
CKPT_COUNT=$(( MAX_STEPS / SAVE_STEPS + 1 ))
CKPT_GB=$(( CKPT_COUNT * 73 ))
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
DISK_WARN=""
[ "$AVAIL_GB" -lt "$CKPT_GB" ] && DISK_WARN="  ⚠️  磁盘可能不够！"

# ---- 启动摘要 ----
cat <<EOF
============================================================
 配置        : $CONFIG
 数据清单    : $DATA_LIST
    $(paste -sd' ; ' "$DATA_LIST" | cut -c1-100)
 输出目录    : $OUTPUT_DIR
 步数        : $MAX_STEPS（每 $SAVE_STEPS 步存档）→ 约 $CKPT_COUNT 个 checkpoint ≈ ${CKPT_GB}GB（磁盘剩余 ${AVAIL_GB}GB）$DISK_WARN
 batch       : micro=$MICRO_BATCH  accum=$GRAD_ACCUM  global=$GLOBAL_BATCH
 预计耗时    : 约 $(awk "BEGIN{printf \"%.1f\", $MAX_STEPS*16.3/3600}") 小时（实测 16.3s/步，不含保存开销）
$(for kv in "LR=$LR" "OPTIMIZER=$OPTIMIZER" "FREEZE_VISION_ENCODER=$FREEZE_VISION_ENCODER" "TRAIN_EXPERT_ONLY=$TRAIN_EXPERT_ONLY" "ENABLE_GRAD_CKPT=$ENABLE_GRAD_CKPT" "NUM_WORKERS=$NUM_WORKERS" "ENABLE_RESUME=$ENABLE_RESUME" "USE_WANDB=$USE_WANDB"; do [ -n "${kv#*=}" ] && echo " 覆盖        : $kv"; done)
============================================================
EOF

# ---- 启动 ----
if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY_RUN] 将要执行的命令："
  echo "cd $REPO_ROOT && bash train.sh tasks/vla/train_lingbotvla.py $CONFIG ${ARGS[*]} $EXTRA_ARGS"
  exit 0
fi

export TOKENIZERS_PARALLELISM=false
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

# 日志双写：train.sh 内部写 ./log.txt（会被下次运行覆盖），这里再存一份到输出目录（永久保留）
bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG" "${ARGS[@]}" ${EXTRA_ARGS:-} 2>&1 | tee "$OUTPUT_DIR/train.log"

echo "日志已保存: $OUTPUT_DIR/train.log"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR/runs --port 6006"
