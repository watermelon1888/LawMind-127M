#!/usr/bin/env bash

set -euo pipefail

source /etc/network_turbo
trap 'unset http_proxy https_proxy HF_HUB_DISABLE_XET' EXIT
export HF_HUB_DISABLE_XET=1

cd /root/autodl-tmp

for batch in $(seq 1 22); do
  PYTHONPATH=/root/autodl-tmp /root/miniconda3/envs/minimind/bin/python -u \
    -m minimind.dataset.evaluate_query_sft_full_retrieval \
    --device cuda \
    --batch-index "$batch" \
    --batch-size 25
done

PYTHONPATH=/root/autodl-tmp /root/miniconda3/envs/minimind/bin/python -u \
  -m minimind.dataset.evaluate_query_sft_full_retrieval \
  --finalize \
  --batch-size 25
