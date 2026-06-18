#!/bin/bash
# VCD / M3ID baselines on LLaVA CHAIR + AMBER.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

python experiments/baselines/vcd_m3id_llava.py \
  --model_path "${MODEL_LLAVA}" \
  --benchmark chair \
  --method vcd \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT_ROOT}/baselines/llava_vcd_chair"

echo "=== LLaVA VCD CHAIR done ==="
