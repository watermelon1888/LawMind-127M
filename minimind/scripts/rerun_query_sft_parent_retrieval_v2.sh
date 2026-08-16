#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_ROOT="/root/autodl-tmp"
RAG_ROOT="$AUTODL_ROOT/rag"
MINIMIND_ROOT="$AUTODL_ROOT/minimind"
WORK_ROOT="$AUTODL_ROOT/minimind-work"
V1_ROOT="$WORK_ROOT/runs/query-sft-parent-evaluation-v1"
V2_ROOT="$WORK_ROOT/runs/query-sft-parent-evaluation-v2"
STATUS="$V2_ROOT/evaluation-status.tsv"
LOG="$V2_ROOT/evaluation-batch.log"
ROLES=(stage_b_control cpt_250m cpt_750m cpt_2b cpt_final)

source /root/miniconda3/etc/profile.d/conda.sh
export PYTHONPATH="$AUTODL_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

stage_dir="$V2_ROOT/candidates/stage_b_control-seed-42/evaluation"
test -s "$stage_dir/summary.json"
test ! -e "$STATUS"
printf 'role\tstatus\tprotocol_legal_rate\tcandidate_pool_complete_hit_at_20\trerank_complete_hit_at_5\tpackaged_complete_hit\n' > "$STATUS"

for index in "${!ROLES[@]}"; do
  role="${ROLES[$index]}"
  inference="$V1_ROOT/candidates/${role}-seed-42/inference/manifest.json"
  output="$V2_ROOT/candidates/${role}-seed-42/evaluation"
  if [[ "$role" != "stage_b_control" ]]; then
    test ! -e "$output"
    mkdir -p "$(dirname "$output")"
    echo "[Query-SFT retrieval v2] Starting $((index + 1))/5: $role" | tee -a "$LOG"
    conda run --no-capture-output -n minimind \
      python -m rag.eval.query_sft_model_evaluation \
      --inference-manifest "$inference" \
      --eval-set "$RAG_ROOT/eval/eval_set.jsonl" \
      --article-index "$RAG_ROOT/chunk/article_index.jsonl" \
      --artifact-dir "$RAG_ROOT/retrieval/artifacts" \
      --tokenizer-path "$MINIMIND_ROOT/model" \
      --output-dir "$output" \
      --device cuda \
      2>&1 | tee "$V2_ROOT/candidates/${role}-seed-42/evaluation.log" | tee -a "$LOG"
  fi
  test -s "$output/summary.json"
  values="$(conda run -n minimind python -c "import json; s=json.load(open('$output/summary.json',encoding='utf-8')); print(s['query_enhancement']['protocol_legal_rate'], s['retrieval']['candidate_pool']['complete_hit'], s['retrieval']['reranked_top5']['complete_hit'], s['packaging']['required_gt']['complete_hit'])")"
  read -r legal pool top5 packaged <<< "$values"
  printf '%s\tcomplete\t%s\t%s\t%s\t%s\n' "$role" "$legal" "$pool" "$top5" "$packaged" >> "$STATUS"
  echo "[Query-SFT retrieval v2] Completed: $role pool=$pool top5=$top5 packaged=$packaged" | tee -a "$LOG"
done

echo "[Query-SFT retrieval v2] All five candidates completed" | tee -a "$LOG"
