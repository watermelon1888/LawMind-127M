#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "用法: bash scripts/run_stage_b_pretrain.sh <获胜run_name> <peak_lr> <stop_tokens>" >&2
  exit 2
fi

RUN_NAME="$1"
PEAK_LR="$2"
STOP_TOKENS="$3"
if [[ ! "$STOP_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "stop_tokens 必须是正整数: $STOP_TOKENS" >&2
  exit 2
fi

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="/root/autodl-tmp/minimind"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
LOG_PATH="$WORK_ROOT/runs/${RUN_NAME}-to-${STOP_TOKENS}.log"

finalize() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  sync
  /usr/bin/shutdown -h now
  exit "$exit_code"
}
trap finalize EXIT INT TERM

source "$WORK_ROOT/env.sh"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate minimind
cd "$PROJECT_ROOT"
mkdir -p "$WORK_ROOT/runs"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "[阶段 B] 恢复 $RUN_NAME，目标 stop_tokens=$STOP_TOKENS" | tee -a "$LOG_PATH"
python -m trainer.train_pretrain \
  --run_name "$RUN_NAME" \
  --peak_lr "$PEAK_LR" \
  --stop_tokens "$STOP_TOKENS" \
  --fixed_full_tokens "$STOP_TOKENS" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --accumulation_steps "$ACCUMULATION_STEPS" \
  --resume \
  --swanlab \
  2>&1 | tee -a "$LOG_PATH"
