#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="/root/autodl-tmp/minimind"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-16}"
LOG_PATH="$WORK_ROOT/runs/stage-b-lr-sweep.log"

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

candidate_specs=(
  "stage-b-lr-3e-4:3e-4"
  "stage-b-lr-5e-4:5e-4"
  "stage-b-lr-7e-4:7e-4"
)

for candidate_spec in "${candidate_specs[@]}"; do
  IFS=: read -r run_name peak_lr <<< "$candidate_spec"
  echo "[阶段 B] 启动 $run_name，peak_lr=$peak_lr" | tee -a "$LOG_PATH"
  python -m trainer.train_pretrain \
    --run_name "$run_name" \
    --peak_lr "$peak_lr" \
    --stop_tokens 80000000 \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --quick_interval_tokens 0 \
    --full_interval_tokens 0 \
    --fixed_quick_tokens 20000000,50000000 \
    --fixed_full_tokens 80000000 \
    --swanlab \
    2>&1 | tee -a "$LOG_PATH"
done
