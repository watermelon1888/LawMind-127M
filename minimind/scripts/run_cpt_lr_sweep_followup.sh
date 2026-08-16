#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="/root/autodl-tmp/minimind"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
LOG_PATH="$WORK_ROOT/runs/cpt/cpt-lr-sweep-followup-r5.log"

source "$WORK_ROOT/env.sh"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate minimind
cd "$PROJECT_ROOT"
mkdir -p "$WORK_ROOT/runs/cpt"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

candidate_specs=(
  "cpt-lr-sweep-r5-3e-4:3e-4"
  "cpt-lr-sweep-r5-4e-4:4e-4"
)

for candidate_spec in "${candidate_specs[@]}"; do
  IFS=: read -r run_name peak_lr <<< "$candidate_spec"
  echo "[CPT] 启动补测 $run_name，peak_lr=$peak_lr" | tee -a "$LOG_PATH"
  python -m trainer.train_cpt \
    --run_name "$run_name" \
    --peak_lr "$peak_lr" \
    --stop_tokens 100073472 \
    --warmup_tokens 25000000 \
    --schedule_tokens 2447821056 \
    --npc_repeat 5 \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --quick_interval_tokens 0 \
    --full_interval_tokens 0 \
    --swanlab \
    2>&1 | tee -a "$LOG_PATH"
done

sync
echo "[CPT] 两档补充 LR sweep 已完成；脚本不会自动关机。" | tee -a "$LOG_PATH"
