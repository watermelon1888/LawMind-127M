#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -gt 1 ]] || [[ $# -eq 1 && "$1" != "--resume" ]]; then
  echo "用法: bash scripts/run_cpt_pretrain.sh [--resume]" >&2
  exit 2
fi

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="/root/autodl-tmp/minimind"
RUN_NAME="cpt-legal-r5-lr-2e-4"
PEAK_LR="2e-4"
STOP_TOKENS="2447821056"
WARMUP_TOKENS="25000000"
SCHEDULE_TOKENS="2447821056"
NPC_REPEAT="5"
FULL_MILESTONES="250085376,500170752,750059520,1000144896,1250033664,1500119040,1750007808,2000093184,2249981952,2447821056"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
LOG_PATH="$WORK_ROOT/runs/cpt/${RUN_NAME}-to-${STOP_TOKENS}.log"
resume_args=()
if [[ $# -eq 1 ]]; then
  resume_args=(--resume)
fi

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
cd "$PROJECT_ROOT"
mkdir -p "$WORK_ROOT/runs/cpt"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "[CPT] 启动正式法律持续训练：run=$RUN_NAME，stop_tokens=$STOP_TOKENS，resume=${resume_args[*]:-false}" | tee -a "$LOG_PATH"
conda run --no-capture-output -n minimind python -m trainer.train_cpt \
  --run_name "$RUN_NAME" \
  --peak_lr "$PEAK_LR" \
  --stop_tokens "$STOP_TOKENS" \
  --warmup_tokens "$WARMUP_TOKENS" \
  --schedule_tokens "$SCHEDULE_TOKENS" \
  --npc_repeat "$NPC_REPEAT" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --accumulation_steps "$ACCUMULATION_STEPS" \
  --quick_interval_tokens 100000000 \
  --full_interval_tokens 0 \
  --fixed_full_tokens "$FULL_MILESTONES" \
  --swanlab \
  "${resume_args[@]}" \
  2>&1 | tee -a "$LOG_PATH"
