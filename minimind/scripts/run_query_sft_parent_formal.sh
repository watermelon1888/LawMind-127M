#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${MINIMIND_PROJECT_ROOT:-/root/autodl-tmp/minimind}"
WORK_ROOT="${MINIMIND_WORK_ROOT:-/root/autodl-tmp/minimind-work}"
RUN_ROOT="${QUERY_SFT_RUN_ROOT:-$WORK_ROOT/runs/query-sft-parent-formal-v1}"
RELEASE_ROOT="$PROJECT_ROOT/dataset/QUERY-POOL/full/query-sft-v2-training-release-r2"
RELEASE_MANIFEST="$RELEASE_ROOT/query-sft-training-release.json"
CANDIDATE="$RELEASE_ROOT/query-sft-v2-training-candidate.jsonl"
EVALUATION_MANIFEST="$PROJECT_ROOT/dataset/RAG-SFT/manifests/evaluation-exclusions-project-rag-v2.json"
TOKENIZER="$PROJECT_ROOT/model"
PARENT_ROOT="$WORK_ROOT/runs/sft-base-parent-formal-v1/candidates"
STATUS="$RUN_ROOT/formal-training-status.tsv"
LOG="$RUN_ROOT/formal-training-batch.log"

ROLES=(stage_b_control cpt_250m cpt_750m cpt_2b cpt_final)
PARENT_HASHES=(
  5dd23163f77db097940115e245ffe41691f0e7380778c897b203c8db91fb2b55
  b8f8c57dbc8a6506fdd953043143932cb367e06d3d9aee29981d1fdb92cc0adf
  2cdd8db9c6a9bef0acf0f529f0aac0fecb8a2b49e81afda2ce08aee266e1bc6f
  a170215114153ea58e51c1a5ac17e2971970a4ce9b437bc5787c4b2e3ca665dd
  6c3030e49efe36a1da47bbbe4fe94b15fc827c2943dbe4b87b1846381ad19283
)

source /root/miniconda3/etc/profile.d/conda.sh
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH=/root/autodl-tmp:/root/autodl-tmp/minimind
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT"

if [[ -e "$STATUS" ]]; then
  echo "Formal status already exists: $STATUS" >&2
  exit 1
fi

for index in "${!ROLES[@]}"; do
  role="${ROLES[$index]}"
  parent="$PARENT_ROOT/${role}-seed-42/training/checkpoints/weights/legal-sft-13128777.pth"
  expected_hash="${PARENT_HASHES[$index]}"
  candidate_root="$RUN_ROOT/candidates/${role}-seed-42"
  actual_hash="$(sha256sum "$parent" | awk '{print $1}')"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    echo "Parent weight SHA256 mismatch: $role" >&2
    exit 1
  fi
  if [[ -e "$candidate_root" ]]; then
    echo "Candidate output already exists: $candidate_root" >&2
    exit 1
  fi
done

printf 'role\tstatus\tcompleted_assistant_tokens\toptimizer_steps\tmodel_only_sha256\n' > "$STATUS"
echo "[Query-SFT formal] Starting five parent runs: 4 epochs, seed=42, SwanLab enabled" | tee -a "$LOG"

for index in "${!ROLES[@]}"; do
  role="${ROLES[$index]}"
  parent="$PARENT_ROOT/${role}-seed-42/training/checkpoints/weights/legal-sft-13128777.pth"
  parent_hash="${PARENT_HASHES[$index]}"
  candidate_root="$RUN_ROOT/candidates/${role}-seed-42"
  run_dir="$candidate_root/run"
  checkpoint_dir="$candidate_root/checkpoints"
  run_name="query-sft-v2-r2-${role}-seed-42"
  candidate_log="$candidate_root/training.log"

  mkdir -p "$candidate_root"
  echo "[Query-SFT formal] Starting $((index + 1))/5: $role" | tee -a "$LOG"
  set +e
  conda run --no-capture-output -n minimind \
    python -m minimind.trainer.train_query_sft \
    --run_name "$run_name" \
    --run_dir "$run_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --release_manifest "$RELEASE_MANIFEST" \
    --candidate_path "$CANDIDATE" \
    --evaluation_manifest "$EVALUATION_MANIFEST" \
    --tokenizer_path "$TOKENIZER" \
    --parent_weights "$parent" \
    --parent_sha256 "$parent_hash" \
    --peak_lr 1.2e-5 \
    --stop_assistant_tokens 170160 \
    --warmup_assistant_tokens 17016 \
    --schedule_assistant_tokens 170160 \
    --floor_ratio 0.1 \
    --micro_batch_size 8 \
    --accumulation_steps 2 \
    --grad_clip 1.0 \
    --log_interval_tokens 4254 \
    --checkpoint_interval_tokens 42540 \
    --device cuda:0 \
    --dtype bfloat16 \
    --num_workers 4 \
    --seed 42 \
    --swanlab \
    --swanlab_project minimind+query \
    --swanlab_workspace Bigwatermelon \
    2>&1 | tee "$candidate_log" | tee -a "$LOG"
  command_status=${PIPESTATUS[0]}
  set -e

  if [[ "$command_status" -ne 0 ]]; then
    printf '%s\tfailed\t0\t0\t-\n' "$role" >> "$STATUS"
    echo "[Query-SFT formal] Training command failed: $role" | tee -a "$LOG"
    exit "$command_status"
  fi
  if grep -q 'SwanLab 初始化失败' "$candidate_log"; then
    printf '%s\tswanlab_failed\t0\t0\t-\n' "$role" >> "$STATUS"
    echo "[Query-SFT formal] SwanLab initialization failed: $role" | tee -a "$LOG"
    exit 1
  fi

  metrics="$run_dir/metrics.jsonl"
  weights="$checkpoint_dir/weights/query-sft-170160.pth"
  if ! grep -q '"completed_assistant_tokens": 170160' "$metrics" || \
     ! grep -q '"weights_exported": true' "$metrics" || \
     [[ ! -f "$weights" ]] || [[ ! -d "$run_dir/swanlog" ]]; then
    printf '%s\tincomplete\t0\t0\t-\n' "$role" >> "$STATUS"
    echo "[Query-SFT formal] Completion validation failed: $role" | tee -a "$LOG"
    exit 1
  fi
  weight_hash="$(sha256sum "$weights" | awk '{print $1}')"
  optimizer_steps="$(grep '"type": "run_complete"' "$metrics" | tail -1 | sed -E 's/.*"optimizer_step": ([0-9]+).*/\1/')"
  printf '%s\tcomplete\t170160\t%s\t%s\n' "$role" "$optimizer_steps" "$weight_hash" >> "$STATUS"
  echo "[Query-SFT formal] Completed: $role, weights=$weight_hash" | tee -a "$LOG"
done

echo "[Query-SFT formal] All five parent runs completed" | tee -a "$LOG"
