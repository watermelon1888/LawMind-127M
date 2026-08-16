#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
MANIFEST_ROOT="${BASE_LR_CALIBRATION_MANIFEST_ROOT:-$WORK_ROOT/manifests/sft-base-lr-calibration-v1}"
RUN_ROOT="${BASE_LR_CALIBRATION_RUN_ROOT:-$WORK_ROOT/runs/sft-base-lr-calibration-v1}"
PLAN="${BASE_LR_CALIBRATION_PLAN:-$MANIFEST_ROOT/calibration-plan.json}"
SUMMARY="$RUN_ROOT/calibration-summary.json"
LOG="$RUN_ROOT/calibration-batch.log"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT"

echo "[基础法律 LR 校准] 开始两个 baseline 和六个独立 trial" | tee -a "$LOG"
echo "[基础法律 LR 校准] 每个 trial 都从原始 CPT model-only 权重启动" | tee -a "$LOG"
echo "[基础法律 LR 校准] 校准 checkpoint 只允许同 trial 恢复" | tee -a "$LOG"
echo "[基础法律 LR 校准] 六个训练 trial 启用 SwanLab，纯评估 baseline 不创建实验" | tee -a "$LOG"

for observation_index in 1 2 3 4 5 6 7 8; do
  echo "[基础法律 LR 校准] 执行观察项 $observation_index/8" | tee -a "$LOG"
  set +e
  PYTHONPATH=/root/autodl-tmp \
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_lr_calibration_runner \
    --plan "$PLAN" \
    --project-root "$PROJECT_ROOT" \
    2>&1 | tee -a "$LOG"
  command_status=${PIPESTATUS[0]}
  set -e
  if [[ "$command_status" -ne 0 ]]; then
    echo "[基础法律 LR 校准] 当前观察项失败，已停止批处理" | tee -a "$LOG"
    break
  fi
done

completed_trials=0
completed_baselines=0
if [[ -d "$RUN_ROOT/trials" ]]; then
  completed_trials=$(find "$RUN_ROOT/trials" -type f -name result.json | wc -l)
fi
if [[ -d "$RUN_ROOT/baselines" ]]; then
  completed_baselines=$(find "$RUN_ROOT/baselines" -maxdepth 1 -type f -name '*.json' | wc -l)
fi
echo "[基础法律 LR 校准] baseline=$completed_baselines/2 trial=$completed_trials/6" | tee -a "$LOG"

if [[ "$completed_baselines" -eq 2 && "$completed_trials" -eq 6 ]]; then
  if [[ ! -e "$SUMMARY" && ! -e "${SUMMARY%.json}.sha256" ]]; then
    PYTHONPATH=/root/autodl-tmp \
    conda run --no-capture-output -n minimind \
      python -m trainer.sft_base_lr_calibration summarize \
      --plan "$PLAN" \
      --run-root "$RUN_ROOT" \
      --output "$SUMMARY" \
      2>&1 | tee -a "$LOG"
  fi
  (
    cd "$RUN_ROOT"
    sha256sum -c calibration-summary.sha256
  )
  echo "[基础法律 LR 校准] 六 trial 完成，等待人工生成审核和共同 LR 决策" | tee -a "$LOG"
else
  echo "[基础法律 LR 校准] 结果尚未完整，可再次运行本脚本严格恢复" | tee -a "$LOG"
fi
