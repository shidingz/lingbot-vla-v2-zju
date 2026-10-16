#!/usr/bin/env bash
# Controlled full-model comparison: identical initialization, data and LR horizon.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_ROOT="${BENCH_ROOT:-/cache/lingbot/profiling/moe_ab_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
BENCH_STEPS="${BENCH_STEPS:-40}"
mkdir -p "$BENCH_ROOT"
read -r -a backends <<< "${BENCH_BACKENDS:-torch gmm}"
for implementation in "${backends[@]}"; do
  [[ "$implementation" == torch || "$implementation" == gmm ]] || { echo "Unknown backend: $implementation" >&2; exit 1; }
  case_root="$BENCH_ROOT/$implementation"
  mkdir -p "$case_root"
  printf '%s\n' "$implementation" > "$case_root/moe_backend.txt"
  # Bash can resume reading its source after torchrun exits; keep that source
  # immutable even while the working copy is being edited.
  cp train_robotwin_npu.sh "$case_root/train_robotwin_npu.snapshot.sh"
  cp tools/profile_robotwin_npu.sh "$case_root/profile_robotwin_npu.snapshot.sh"
  # An empty rank list runs the same bounded performance study without recording
  # any trace. This excludes profiler export/collection from the speed comparison.
  extra=(--train.profile_ranks "")
  if [[ "$implementation" == gmm && "${BENCH_PROFILE_GMM:-0}" == 1 ]]; then
    extra=(--train.profile_ranks 0 --train.profile_start_step 31 --train.profile_end_step 33)
  fi
  status=0
  RUN_NAME="moe_ab_$implementation" PROFILE_ROOT="$case_root" \
    STOP_AFTER_STEPS="$BENCH_STEPS" MASTER_PORT="${MASTER_PORT:-62518}" \
    TRAIN_SCRIPT="$case_root/train_robotwin_npu.snapshot.sh" \
    LINGBOT_NPU_MOE_IMPL="$implementation" \
    bash tools/profile_robotwin_npu.sh "${extra[@]}" > "$case_root/console.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$case_root/exit_code"
  if ((status != 0)); then
    tail -60 "$case_root/console.log" >&2
    exit "$status"
  fi
done
printf 'Completed comparison: %s\n' "$BENCH_ROOT"
