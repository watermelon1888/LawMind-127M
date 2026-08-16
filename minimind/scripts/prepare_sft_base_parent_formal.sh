#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
CALIBRATION_MANIFEST_ROOT="${BASE_LR_CALIBRATION_MANIFEST_ROOT:-$WORK_ROOT/manifests/sft-base-lr-calibration-v1}"
CALIBRATION_RUN_ROOT="${BASE_LR_CALIBRATION_RUN_ROOT:-$WORK_ROOT/runs/sft-base-lr-calibration-v1}"
MANIFEST_ROOT="${BASE_PARENT_FORMAL_MANIFEST_ROOT:-$WORK_ROOT/manifests/sft-base-parent-formal-v1}"
RUN_ROOT="${BASE_PARENT_FORMAL_RUN_ROOT:-$WORK_ROOT/runs/sft-base-parent-formal-v1}"

CALIBRATION_SUMMARY="$CALIBRATION_RUN_ROOT/calibration-summary.json"
DECISION="$CALIBRATION_MANIFEST_ROOT/calibration-decision.json"
SPEC="$MANIFEST_ROOT/formal-parent-spec.json"
PLAN="$MANIFEST_ROOT/formal-parent-plan.json"

STAGE_B_CONTROL="${STAGE_B_CONTROL:-$WORK_ROOT/checkpoints/stage-b/stage-b-lr-1e-3/weights/pretrain-9737362944.pth}"
CPT_ROOT="${CPT_ROOT:-$WORK_ROOT/checkpoints/cpt/cpt-legal-r5-lr-2e-4/weights}"
CPT_250M="${CPT_250M:-$CPT_ROOT/cpt-250085376.pth}"
CPT_750M="${CPT_750M:-$CPT_ROOT/cpt-750059520.pth}"
CPT_2B="${CPT_2B:-$CPT_ROOT/cpt-2000093184.pth}"
CPT_FINAL="${CPT_FINAL:-$CPT_ROOT/cpt-2447821056.pth}"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[基础法律正式父权重] 开始 CPU 资产预检与五候选计划发布"
echo "[基础法律正式父权重] 本入口不加载 CUDA、不启动训练"
echo "[基础法律正式父权重] 校准 checkpoint 不得作为正式父权重"

if [[ ! -e "$DECISION" && ! -e "${DECISION%.json}.sha256" ]]; then
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_lr_calibration decide \
    --summary "$CALIBRATION_SUMMARY" \
    --selected-peak-lr 1.2e-5 \
    --output "$DECISION"
fi

if [[ ! -e "$SPEC" && ! -e "${SPEC%.json}.sha256" ]]; then
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_parent_evaluation spec \
    --decision "$DECISION" \
    --stage-b-control "$STAGE_B_CONTROL" \
    --cpt-250m "$CPT_250M" \
    --cpt-750m "$CPT_750M" \
    --cpt-2b "$CPT_2B" \
    --cpt-final "$CPT_FINAL" \
    --output "$SPEC"
fi

if [[ ! -e "$PLAN" && ! -e "${PLAN%.json}.sha256" ]]; then
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_parent_evaluation prepare \
    --spec "$SPEC" \
    --output-root "$RUN_ROOT" \
    --output "$PLAN"
fi

(
  cd "$CALIBRATION_MANIFEST_ROOT"
  sha256sum -c calibration-decision.sha256
)
(
  cd "$MANIFEST_ROOT"
  sha256sum -c formal-parent-spec.sha256 formal-parent-plan.sha256
)

echo "[基础法律正式父权重] CPU 执行准备完成: $PLAN"
echo "[基础法律正式父权重] 尚未执行五候选 baseline、训练或生成评估"
