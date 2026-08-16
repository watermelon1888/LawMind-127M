#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_ROOT="/root/autodl-tmp"
MINIMIND_ROOT="$AUTODL_ROOT/minimind"
RAG_ROOT="$AUTODL_ROOT/rag"
WORK_ROOT="$AUTODL_ROOT/minimind-work"
TRAIN_ROOT="$WORK_ROOT/runs/query-sft-parent-formal-v2"
OUTPUT_ROOT="$WORK_ROOT/runs/query-sft-parent-evaluation-v1"
QUERY_INPUTS="$WORK_ROOT/query-inputs-140.jsonl"
STATUS="$OUTPUT_ROOT/evaluation-status.tsv"
LOG="$OUTPUT_ROOT/evaluation-batch.log"

ROLES=(stage_b_control cpt_250m cpt_750m cpt_2b cpt_final)
WEIGHT_HASHES=(
  ecc7f6701b6dec4109e9898279be7efed2ecfd920f97d3b21a917eb1955d6a8d
  5014368ee72869d4f68dc31d321805f8f87e6165260504dbad331f360d5864e1
  567de82e62c3e1c16bc796a28e1c7a3d7bd629b1375c6bf8fafebb70d686cb32
  09c95990c19a41182d40eaabbb12d6285ca5f3e64601a25d0ff02c5e5243d4a1
  62a6718abac062e7997daa2b1ccbc8d49f50bd1da2d8fae30fa137a3127c1450
)

source /root/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="$AUTODL_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Evaluation output already exists: $OUTPUT_ROOT" >&2
  exit 1
fi
if [[ "$(wc -l < "$QUERY_INPUTS")" -ne 140 ]]; then
  echo "Query input record count is not 140" >&2
  exit 1
fi

for index in "${!ROLES[@]}"; do
  role="${ROLES[$index]}"
  weights="$TRAIN_ROOT/candidates/${role}-seed-42/checkpoints/weights/query-sft-170160.pth"
  actual_hash="$(sha256sum "$weights" | awk '{print $1}')"
  if [[ "$actual_hash" != "${WEIGHT_HASHES[$index]}" ]]; then
    echo "Query-SFT weight SHA256 mismatch: $role" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
printf 'role\tstatus\tinference_applied\tinference_fallback\tprotocol_legal_rate\trerank_complete_hit_at_5\tpackaged_complete_hit\n' > "$STATUS"
echo "[Query-SFT evaluation] Starting five formal candidates" | tee -a "$LOG"

for index in "${!ROLES[@]}"; do
  role="${ROLES[$index]}"
  weights="$TRAIN_ROOT/candidates/${role}-seed-42/checkpoints/weights/query-sft-170160.pth"
  candidate_root="$OUTPUT_ROOT/candidates/${role}-seed-42"
  inference_dir="$candidate_root/inference"
  evaluation_dir="$candidate_root/evaluation"
  mkdir -p "$candidate_root"

  echo "[Query-SFT evaluation] Inference $((index + 1))/5: $role" | tee -a "$LOG"
  conda run --no-capture-output -n minimind \
    python -m minimind.trainer.query_sft_inference \
    --input_path "$QUERY_INPUTS" \
    --weights_path "$weights" \
    --tokenizer_path "$MINIMIND_ROOT/model" \
    --output_dir "$inference_dir" \
    --device cuda:0 \
    2>&1 | tee "$candidate_root/inference.log" | tee -a "$LOG"

  echo "[Query-SFT evaluation] Retrieval $((index + 1))/5: $role" | tee -a "$LOG"
  conda run --no-capture-output -n minimind \
    python -m rag.eval.query_sft_model_evaluation \
    --inference-manifest "$inference_dir/manifest.json" \
    --eval-set "$RAG_ROOT/eval/eval_set.jsonl" \
    --article-index "$RAG_ROOT/chunk/article_index.jsonl" \
    --artifact-dir "$RAG_ROOT/retrieval/artifacts" \
    --tokenizer-path "$MINIMIND_ROOT/model" \
    --output-dir "$evaluation_dir" \
    --device cuda \
    2>&1 | tee "$candidate_root/evaluation.log" | tee -a "$LOG"

  summary="$evaluation_dir/summary.json"
  manifest="$inference_dir/manifest.json"
  test -s "$inference_dir/records.jsonl"
  test -s "$evaluation_dir/enhanced-records.jsonl"
  test -s "$evaluation_dir/report.md"
  test -s "$summary"
  test -s "$evaluation_dir/manifest.json"

  values="$(conda run -n minimind python -c "import json; i=json.load(open('$manifest',encoding='utf-8')); s=json.load(open('$summary',encoding='utf-8')); print(i['records']['applied'], i['records']['fallback'], s['query_enhancement']['protocol_legal_rate'], s['retrieval']['reranked_top5']['complete_hit'], s['packaging']['required_gt']['complete_hit'])")"
  read -r applied fallback legal top5 packaged <<< "$values"
  printf '%s\tcomplete\t%s\t%s\t%s\t%s\t%s\n' "$role" "$applied" "$fallback" "$legal" "$top5" "$packaged" >> "$STATUS"
  echo "[Query-SFT evaluation] Completed: $role top5=$top5 packaged=$packaged" | tee -a "$LOG"
done

echo "[Query-SFT evaluation] All five candidates completed" | tee -a "$LOG"
