#!/bin/bash
# TVER vs EIC ablation on LLaVA CHAIR.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

OUT="${OUT_ROOT}/ablation_tver"
mkdir -p "${OUT}"

for SCRIPT in tver_head_select_chair tver_signal_chair; do
  python "experiments/ablations/${SCRIPT}.py" \
      --seed 3407 \
      --model_path "${MODEL_LLAVA}" \
      --data_path "${COCO_DIR}/val2014" \
      --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
      --layer_index 1 \
      --alpha 0.7 \
      --out_path "${OUT}/${SCRIPT}" \
      --num_eval_samples 500
  run_chair_metrics "${OUT}/${SCRIPT}/${SCRIPT%.py}.jsonl" "${OUT}/${SCRIPT}_results.json"
done

echo "=== TVER ablation done ==="
