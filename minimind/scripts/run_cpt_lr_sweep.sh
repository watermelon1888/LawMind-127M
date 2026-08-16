#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="/root/autodl-tmp/minimind"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
LOG_PATH="$WORK_ROOT/runs/cpt/cpt-lr-sweep-r5.log"

source "$WORK_ROOT/env.sh"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate minimind
cd "$PROJECT_ROOT"
mkdir -p "$WORK_ROOT/runs/cpt"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

candidate_specs=(
  "cpt-lr-sweep-r5-5e-5:5e-5"
  "cpt-lr-sweep-r5-1e-4:1e-4"
  "cpt-lr-sweep-r5-2e-4:2e-4"
)

for candidate_spec in "${candidate_specs[@]}"; do
  IFS=: read -r run_name peak_lr <<< "$candidate_spec"
  echo "[CPT] 启动 $run_name，peak_lr=$peak_lr" | tee -a "$LOG_PATH"
  python -m trainer.train_cpt \
    --run_name "$run_name" \
    --peak_lr "$peak_lr" \
    --stop_tokens 200146944 \
    --warmup_tokens 25000000 \
    --schedule_tokens 2447821056 \
    --npc_repeat 5 \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --quick_interval_tokens 100000000 \
    --full_interval_tokens 0 \
    --swanlab \
    2>&1 | tee -a "$LOG_PATH"
done

sync
echo "[CPT] 三档 LR sweep 已完成；脚本不会自动关机。" | tee -a "$LOG_PATH"
