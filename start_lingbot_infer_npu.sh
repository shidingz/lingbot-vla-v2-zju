#!/usr/bin/env bash
# =============================================================================
# LingBot-VLA v2 — 多卡 NPU 推理服务启动器（Huawei ModelArts / Ascend 910B）
#
# 在选定的每张 NPU 上起 num_per_card 个常驻 websocket policy server，
# 端口从 start_port 逐个递增；slot s -> 卡 cards[s % n] , 端口 start_port+s。
# 客户端用 deploy/websocket_client_policy.py 连 127.0.0.1:<port>；
# 冒烟脚本: python /home/ma-user/work/lingbot-infer/smoke_client.py (端口 9330)。
#
# 用法:
#   bash start_lingbot_infer_npu.sh --model_path <hf_ckpt 目录>
#   bash start_lingbot_infer_npu.sh --model_path ... --cards 0,1,2,3 --num_per_card 1
#   bash start_lingbot_infer_npu.sh --model_path ... --use_bf16 True --use_fp32 False
#
# 常用参数:
#   --model_path   必填。训练 run 的 <run>/checkpoints/global_step_N/hf_ckpt
#                  （同级 <run>/lingbotvla_cli.yaml 必须存在）；或 lingbot-infer 的
#                  checkpoints/global_step_0/hf_ckpt（指向基座模型）
#   --cards        用哪些卡（逗号分隔），默认 0,1,2,3,4,5,6,7
#   --num_per_card 每卡 server 数，默认 1（每 server 约 20-24GB 显存）
#   --start_port   起始端口，默认 9330
#   --use_length   action chunk 长度，默认 50
#   --use_bf16 / --use_fp32 / --use_compile   推理精度与编译，默认 True/False/False
#   --log_dir      日志目录，默认 /cache/lingbot/infer_logs/<时间戳>
#
# Ctrl-C 退出会杀掉本次启动的所有 server。PIDs 存在 <log_dir>/inference_pids.txt。
# =============================================================================
set -uo pipefail

REPO_ROOT=/home/ma-user/work/lingbot-vla-v2
CONDA_ENV=PyTorch-2.6.0
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
QWEN3VL_PATH="${QWEN3VL_PATH:-/home/ma-user/work/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master}"

model_path=""
cards="0,1,2,3,4,5,6,7"
num_per_card=1
start_port=9330
use_length=50
use_bf16=True
use_fp32=False
use_compile=False
log_dir=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path)   model_path="$2"; shift 2 ;;
    --cards)        cards="$2"; shift 2 ;;
    --num_per_card) num_per_card="$2"; shift 2 ;;
    --start_port)   start_port="$2"; shift 2 ;;
    --use_length)   use_length="$2"; shift 2 ;;
    --use_bf16)     use_bf16="$2"; shift 2 ;;
    --use_fp32)     use_fp32="$2"; shift 2 ;;
    --use_compile)  use_compile="$2"; shift 2 ;;
    --log_dir)      log_dir="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

[ -n "$model_path" ] || { echo "[错误] 必须提供 --model_path"; exit 1; }
[ -d "$model_path" ] || { echo "[错误] model_path 不存在: $model_path"; exit 1; }
cli_yaml="$(dirname "$(dirname "$(dirname "$model_path")")")/lingbotvla_cli.yaml"
[ -f "$cli_yaml" ] || { echo "[错误] 找不到训练配置 $cli_yaml（推理需要 <run>/lingbotvla_cli.yaml）"; exit 1; }
[ -f "$CONDA_SH" ] || { echo "[错误] 找不到 $CONDA_SH"; exit 1; }

IFS=',' read -ra card_arr <<< "$cards"
n_cards=${#card_arr[@]}
num_slots=$(( n_cards * num_per_card ))
timestamp=$(date +%Y%m%d_%H%M%S)
log_dir="${log_dir:-/cache/lingbot/infer_logs/${timestamp}}"
mkdir -p "$log_dir"
pid_file="$log_dir/inference_pids.txt"
> "$pid_file"

declare -a server_pids=()
cleanup() {
  echo ""
  echo "=== 停止本次启动的推理 server（进程组） ==="
  # server 由 setsid 启动，kill 整个进程组才能连 python 子进程一起停
  for pid in "${server_pids[@]:-}"; do
    [ -z "$pid" ] && continue
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ]; then kill -TERM -- "-$pgid" 2>/dev/null; else kill -TERM "$pid" 2>/dev/null; fi
    echo "  TERM pid=$pid pgid=${pgid:-?}"
  done
  sleep 2
  for pid in "${server_pids[@]:-}"; do
    [ -z "$pid" ] && continue
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  done
  echo "=== 已全部停止 ==="
}
trap cleanup EXIT INT TERM

echo "模型: $model_path"
echo "训练配置: $cli_yaml"
echo "启动 ${num_slots} 个 server（${n_cards} 卡 × ${num_per_card}/卡），端口 ${start_port}~$((start_port + num_slots - 1))"
echo "日志目录: $log_dir"

for slot in $(seq 0 $((num_slots-1))); do
  card=${card_arr[$(( slot % n_cards ))]}
  port=$(( start_port + slot ))
  log_file="$log_dir/slot${slot}_card${card}_port${port}.log"

  # 每进程只可见一张卡：进程内 npu:0 == 物理卡 $card
  ASCEND_RT_VISIBLE_DEVICES="$card" ASCEND_VISIBLE_DEVICES="$card" \
  QWEN3VL_PATH="$QWEN3VL_PATH" \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
  setsid bash -c "source $CONDA_SH && conda activate $CONDA_ENV && cd $REPO_ROOT && \
    ASCEND_RT_VISIBLE_DEVICES=$card ASCEND_VISIBLE_DEVICES=$card QWEN3VL_PATH=$QWEN3VL_PATH \
    python -m deploy.lingbot_vla_v2_policy \
      --model_path '$model_path' \
      --use_length $use_length \
      --use_bf16 $use_bf16 \
      --use_fp32 $use_fp32 \
      --use_compile $use_compile \
      --port $port" > "$log_file" 2>&1 &

  pid=$!
  server_pids+=("$pid")
  echo "$pid" >> "$pid_file"
  echo "  [slot $slot] 卡 $card, 端口 $port, PID $pid, 日志 $log_file"
  sleep 1
done

echo ""
echo "全部 server 已启动。等待就绪（可 tail 日志查看 'Loading' / 'Policy server' 字样）..."
echo "PIDs: $pid_file"

# 常驻；Ctrl-C 统一退出
wait
