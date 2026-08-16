#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
SOURCE_PLAN="${PARENT_EVAL_V3_PLAN:-$WORK_ROOT/manifests/sft-parent-evaluation-first-round-v3/parent-evaluation-plan.json}"
TARGET_PLAN="${PARENT_EVAL_V4_PLAN:-$WORK_ROOT/manifests/sft-parent-evaluation-first-round-v4/parent-evaluation-plan.json}"
SOURCE_RUN_ROOT="${PARENT_V3_RUN_ROOT:-$WORK_ROOT/runs/sft-parent-evaluation-first-round-v3}"
TARGET_RUN_ROOT="${PARENT_V4_RUN_ROOT:-$WORK_ROOT/runs/sft-parent-evaluation-first-round-v4}"
OUTPUT="${PARENT_V3_MIGRATION_REPORT:-$TARGET_RUN_ROOT/v3-result-migration.json}"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[父权重比较] 开始把已完成 v3 trial 审计迁移到 v4"
echo "[父权重比较] v3 运行根目录: $SOURCE_RUN_ROOT"
echo "[父权重比较] v4 运行根目录: $TARGET_RUN_ROOT"

conda run --no-capture-output -n minimind python -m trainer.sft_parent_evaluation migrate-v3 \
  --source-plan "$SOURCE_PLAN" \
  --target-plan "$TARGET_PLAN" \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --target-run-root "$TARGET_RUN_ROOT" \
  --output "$OUTPUT"

(
  cd "$(dirname "$OUTPUT")"
  sha256sum -c "$(basename "${OUTPUT%.json}.sha256")"
)

echo "[父权重比较] v3 trial 迁移完成: $OUTPUT"
