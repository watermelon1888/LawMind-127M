#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
RAG_ROOT="${RAG_PROJECT_ROOT:-/root/autodl-tmp/rag}"
PLAN="${PARENT_EVAL_PLAN:-$WORK_ROOT/manifests/sft-parent-evaluation-first-round-v4/parent-evaluation-plan.json}"
OUTPUT_ROOT="${PARENT_CHAIN_OUTPUT_ROOT:-$WORK_ROOT/runs/sft-parent-evaluation-first-round-v4}"

RAG_FIXED_MANIFEST="${RAG_FIXED_MANIFEST:-$PROJECT_ROOT/dataset/RAG-SFT/manifests/rag-sft-training-v1-final-manifest-768.json}"
RAG_CANDIDATE="${RAG_CANDIDATE:-$PROJECT_ROOT/dataset/RAG-SFT/standardized/768/rag-sft-training-v1-final-candidate.jsonl}"
PAIR_RECORDS="${PAIR_RECORDS:-$RAG_ROOT/eval/results/parent-pair-evaluation-v2-release/pair-evaluation-records.jsonl}"
ARTICLE_INDEX="${ARTICLE_INDEX:-$RAG_ROOT/chunk/article_index.jsonl}"
DEVICE="${PARENT_EVAL_DEVICE:-cuda:0}"

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "[父权重比较] 开始发布五条连续链的 CPU 运行绑定"
echo "[父权重比较] 输出根目录: $OUTPUT_ROOT"
echo "[父权重比较] 本入口复验资产，不加载模型、不启动 GPU"

for role in stage_b_control cpt_250m cpt_750m cpt_2b cpt_final; do
  chain_id="${role}-seed-42"
  conda run --no-capture-output -n minimind python -m trainer.sft_parent_chain_runner prepare \
    --plan "$PLAN" \
    --chain-id "$chain_id" \
    --output-root "$OUTPUT_ROOT" \
    --rag-fixed-manifest "$RAG_FIXED_MANIFEST" \
    --rag-candidate "$RAG_CANDIDATE" \
    --pair-records "$PAIR_RECORDS" \
    --article-index "$ARTICLE_INDEX" \
    --project-root "$PROJECT_ROOT" \
    --device "$DEVICE"
done

for role in stage_b_control cpt_250m cpt_750m cpt_2b cpt_final; do
  chain_id="${role}-seed-42"
  manifest_dir="$OUTPUT_ROOT/chains/$chain_id"
  (
    cd "$manifest_dir"
    sha256sum -c chain-execution-manifest.sha256
  )
done

echo "[父权重比较] 五条连续链运行绑定发布完成"
echo "[父权重比较] 尚未执行任何观察点"
