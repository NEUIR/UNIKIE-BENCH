
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "USE:$0 <MODEL_NAME>"
  exit 1
fi

MODEL="$1"
DATASETS=(Medical-Services) # datasets name
for ds in "${DATASETS[@]}"; do
  echo "==> Running: dataset ${ds} with model ${MODEL}"
  python src/request_openai.py --dataset "${ds}" --model "${MODEL}" --api-key "xxxx" --api-base "xxxx"  
done
