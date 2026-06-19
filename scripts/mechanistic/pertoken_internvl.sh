#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/internvl_eic.pt}"
OUT="${OUT_ROOT}/mechanistic/pertoken_internvl"
mkdir -p "${OUT}"

python experiments/pertoken_analysis_internvl.py \
  --model_path "${MODEL_INTERNVL}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --c_scores_path "${SCORES}" \
  --layer_index 1 \
  --alpha 0.7 \
  --num_eval_samples "${NSAMP:-500}" \
  --out_path "${OUT}"

echo "=== InternVL pertoken analysis done ==="
