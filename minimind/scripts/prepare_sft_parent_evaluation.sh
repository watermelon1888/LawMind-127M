#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$WORK_ROOT/manifests}"
OUTPUT_ROOT="${PARENT_EVAL_OUTPUT_ROOT:-$WORK_ROOT/manifests/sft-parent-evaluation-first-round-v4}"

STAGE_B_CONTROL="${STAGE_B_CONTROL:-$WORK_ROOT/checkpoints/stage-b/stage-b-lr-1e-3/weights/pretrain-9737362944.pth}"
CPT_250M="${CPT_250M:-$WORK_ROOT/checkpoints/cpt/cpt-legal-r5-lr-2e-4/weights/cpt-250085376.pth}"
CPT_750M="${CPT_750M:-$WORK_ROOT/checkpoints/cpt/cpt-legal-r5-lr-2e-4/weights/cpt-750059520.pth}"
CPT_2B="${CPT_2B:-$WORK_ROOT/checkpoints/cpt/cpt-legal-r5-lr-2e-4/weights/cpt-2000093184.pth}"
CPT_FINAL="${CPT_FINAL:-$WORK_ROOT/checkpoints/cpt/cpt-legal-r5-lr-2e-4/weights/cpt-2447821056.pth}"

EVALUATION_EXCLUSIONS="${EVALUATION_EXCLUSIONS:-$PROJECT_ROOT/dataset/RAG-SFT/manifests/evaluation-exclusions-project-rag-v2.json}"
RAG_SFT_RELEASE="${RAG_SFT_RELEASE:-$PROJECT_ROOT/dataset/RAG-SFT/manifests/rag-sft-training-v1-release.json}"
RAG_PAIR_EVALUATION="${RAG_PAIR_EVALUATION:-/root/autodl-tmp/rag/eval/results/parent-pair-evaluation-v2-release/pair-evaluation-manifest.json}"
GENERAL_VALIDATION="${GENERAL_VALIDATION:-$WORK_ROOT/manifests/stage-b-shards-v1.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$PROJECT_ROOT/model}"

SPEC_PATH="$OUTPUT_ROOT/parent-evaluation-spec.json"
PLAN_PATH="$OUTPUT_ROOT/parent-evaluation-plan.json"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"

base_manifest_args=(--manifest_root "$MANIFEST_ROOT")
if [[ -n "${BASE_SFT_MANIFEST:-}" ]]; then
  base_manifest_args=(--base_sft_manifest "$BASE_SFT_MANIFEST")
fi

echo "[父权重比较] 开始 CPU 资产预检与计划发布"
echo "[父权重比较] 输出目录: $OUTPUT_ROOT"
echo "[父权重比较] 第一轮固定为 5 个候选、6 个观察点和 30 个 trial"
echo "[父权重比较] 本入口不加载 CUDA、不启动训练、不自动关机"

conda run --no-capture-output -n minimind python -m trainer.sft_parent_evaluation spec \
  --stage_b_control "$STAGE_B_CONTROL" \
  --cpt_250m "$CPT_250M" \
  --cpt_750m "$CPT_750M" \
  --cpt_2b "$CPT_2B" \
  --cpt_final "$CPT_FINAL" \
  "${base_manifest_args[@]}" \
  --general_validation "$GENERAL_VALIDATION" \
  --evaluation_exclusions "$EVALUATION_EXCLUSIONS" \
  --rag_sft_release "$RAG_SFT_RELEASE" \
  --rag_pair_evaluation "$RAG_PAIR_EVALUATION" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --output "$SPEC_PATH"

conda run --no-capture-output -n minimind python -m trainer.sft_parent_evaluation prepare \
  --spec "$SPEC_PATH" \
  --output "$PLAN_PATH"

(
  cd "$OUTPUT_ROOT"
  sha256sum -c parent-evaluation-spec.sha256 parent-evaluation-plan.sha256
)

echo "[父权重比较] CPU 计划发布完成: $PLAN_PATH"
echo "[父权重比较] 尚未运行五候选 GPU 训练或生成式评估"
