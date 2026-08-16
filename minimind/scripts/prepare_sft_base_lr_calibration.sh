#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
MANIFEST_ROOT="${BASE_LR_CALIBRATION_MANIFEST_ROOT:-$WORK_ROOT/manifests/sft-base-lr-calibration-v1}"
RUN_ROOT="${BASE_LR_CALIBRATION_RUN_ROOT:-$WORK_ROOT/runs/sft-base-lr-calibration-v1}"

DATA_MANIFEST="${BASE_SFT_MANIFEST:-$WORK_ROOT/manifests/disc-law-sft-v1-formal-768.json}"
EVALUATION_MANIFEST="${BASE_SFT_EVALUATION_MANIFEST:-$PROJECT_ROOT/dataset/RAG-SFT/manifests/evaluation-exclusions-project-rag-v2.json}"
GENERAL_VALIDATION="${GENERAL_VALIDATION:-$WORK_ROOT/manifests/stage-b-shards-v1.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/model}"
STAGE_B_CONTROL="${STAGE_B_CONTROL:-$WORK_ROOT/checkpoints/stage-b/stage-b-lr-1e-3/weights/pretrain-9737362944.pth}"
CPT_FINAL="${CPT_FINAL:-$WORK_ROOT/checkpoints/cpt/cpt-legal-r5-lr-2e-4/weights/cpt-2447821056.pth}"

GENERATION_ROOT="$MANIFEST_ROOT/fixed-generation-development-v1"
GENERATION_MANIFEST="$GENERATION_ROOT/manifest.json"
SPEC="$MANIFEST_ROOT/calibration-spec.json"
PLAN="$MANIFEST_ROOT/calibration-plan.json"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[基础法律 LR 校准] 开始 CPU 资产预检与执行计划发布"
echo "[基础法律 LR 校准] 本入口不加载 CUDA、不启动训练"
echo "[基础法律 LR 校准] 校准权重不得接着训练为正式五候选结果"

if [[ ! -e "$GENERATION_MANIFEST" && ! -e "${GENERATION_MANIFEST%.json}.sha256" ]]; then
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_generation_evaluation prepare \
    --data-manifest "$DATA_MANIFEST" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --output-dir "$GENERATION_ROOT"
fi

if [[ ! -e "$SPEC" && ! -e "${SPEC%.json}.sha256" ]]; then
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_lr_calibration spec \
    --stage-b-control "$STAGE_B_CONTROL" \
    --cpt-final "$CPT_FINAL" \
    --data-manifest "$DATA_MANIFEST" \
    --evaluation-manifest "$EVALUATION_MANIFEST" \
    --general-validation "$GENERAL_VALIDATION" \
    --generation-manifest "$GENERATION_MANIFEST" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --output "$SPEC"
fi

if [[ ! -e "$PLAN" && ! -e "${PLAN%.json}.sha256" ]]; then
  conda run --no-capture-output -n minimind \
    python -m trainer.sft_base_lr_calibration prepare \
    --spec "$SPEC" \
    --output-root "$RUN_ROOT" \
    --output "$PLAN"
fi

(
  cd "$GENERATION_ROOT"
  sha256sum -c records.sha256 manifest.sha256
)
(
  cd "$MANIFEST_ROOT"
  sha256sum -c calibration-spec.sha256 calibration-plan.sha256
)

echo "[基础法律 LR 校准] CPU 执行准备完成: $PLAN"
echo "[基础法律 LR 校准] 尚未执行 baseline、GPU 训练或生成评估"
