#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
MANIFEST_ROOT="${BASE_PARENT_FORMAL_MANIFEST_ROOT:-$WORK_ROOT/manifests/sft-base-parent-formal-v1}"
RUN_ROOT="${BASE_PARENT_FORMAL_RUN_ROOT:-$WORK_ROOT/runs/sft-base-parent-formal-v1}"
PLAN="${BASE_PARENT_FORMAL_PLAN:-$MANIFEST_ROOT/formal-parent-plan.json}"
SUMMARY="$RUN_ROOT/formal-parent-summary.json"
LOG="$RUN_ROOT/formal-parent-batch.log"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT"

echo "[基础法律正式父权重] 开始五个原始 CPT 候选的连续完整单遍训练" | tee -a "$LOG"
echo "[基础法律正式父权重] 共同 peak LR=1.2e-5，seed=42" | tee -a "$LOG"
echo "[基础法律正式父权重] 每个候选仅允许恢复自己的正式 run" | tee -a "$LOG"
echo "[基础法律正式父权重] 五条训练链启用 SwanLab，本地 JSONL 始终保留" | tee -a "$LOG"

for candidate_index in 1 2 3 4 5; do
  echo "[基础法律正式父权重] 执行候选 $candidate_index/5" | tee -a "$LOG"
  set +e
  PYTHONPATH=/root/autodl-tmp \
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_parent_evaluation_runner \
    --plan "$PLAN" \
    --project-root "$PROJECT_ROOT" \
    2>&1 | tee -a "$LOG"
  command_status=${PIPESTATUS[0]}
  set -e
  if [[ "$command_status" -ne 0 ]]; then
    echo "[基础法律正式父权重] 当前候选失败，已停止批处理" | tee -a "$LOG"
    break
  fi
done

completed_candidates=0
if [[ -d "$RUN_ROOT/candidates" ]]; then
  completed_candidates=$(find "$RUN_ROOT/candidates" -mindepth 2 -maxdepth 2 -type f -name result.json | wc -l)
fi
echo "[基础法律正式父权重] candidate=$completed_candidates/5" | tee -a "$LOG"

if [[ "$completed_candidates" -eq 5 ]]; then
  if [[ ! -e "$SUMMARY" && ! -e "${SUMMARY%.json}.sha256" ]]; then
    PYTHONPATH=/root/autodl-tmp \
    conda run --no-capture-output -n minimind \
      python -m trainer.sft_base_parent_evaluation summarize \
      --plan "$PLAN" \
      --run-root "$RUN_ROOT" \
      --output "$SUMMARY" \
      2>&1 | tee -a "$LOG"
  fi
  (
    cd "$RUN_ROOT"
    sha256sum -c formal-parent-summary.sha256
  )
  echo "[基础法律正式父权重] 五候选完成，等待匿名生成审核和第一轮排序" | tee -a "$LOG"
else
  echo "[基础法律正式父权重] 结果尚未完整，可再次运行本脚本严格恢复" | tee -a "$LOG"
fi
