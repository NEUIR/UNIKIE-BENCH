#!/usr/bin/env bash
set -euo pipefail
MODELS=(xxx) # models name

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
fi
DATASETS=(xxxxx) # datasets name
for MODEL in "${MODELS[@]}"; do
  echo "==== Evaluating model: ${MODEL} ===="
  for ds in "${DATASETS[@]}"; do
    pred="./results/${ds}/result_${MODEL}.jsonl"
    echo "==> Evaluating: dataset ${ds}, model ${MODEL}"
    if [[ ! -f "$pred" ]]; then
      echo "    Prediction file missing: $pred, skipping"
      continue
    fi
    python src/evaluate_results.py --pred "$pred" --dataset "$ds" 
  done
done