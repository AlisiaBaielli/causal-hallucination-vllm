#!/usr/bin/env bash
# Minimal GPU smoke test: 1-image CHAIR (vanilla + CHALL).
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
if [ ! -f "${SCORES}" ]; then
  echo "Missing ${SCORES}. Run: bash scripts/calibrate/llava.sh"
  exit 1
fi

OUT="${OUT_ROOT}/smoke"
mkdir -p "${OUT}"

python experiments/chair/llava.py \
  --seed 3407 \
  --model_path "${MODEL_LLAVA}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/vanilla" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --no_hook \
  --method_name vanilla

python experiments/chair/llava.py \
  --seed 3407 \
  --model_path "${MODEL_LLAVA}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/chall" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --c_scores_path "${SCORES}" \
  --layer_index 1 \
  --alpha 0.7 \
  --method_name chall

run_chair_metrics "${OUT}/chall/chall.jsonl" "${OUT}/chall/chair.json"
run_chair_metrics "${OUT}/vanilla/vanilla.jsonl" "${OUT}/vanilla/chair.json"
echo "=== Smoke test complete ==="
