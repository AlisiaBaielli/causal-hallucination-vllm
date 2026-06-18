#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

CALIB_JSONL="${COCO_DIR}/calibration.jsonl"
if [ ! -f "${CALIB_JSONL}" ]; then
  python -m causal_core.make_calibration_jsonl \
    --instances "${COCO_DIR}/annotations/instances_val2014.json" \
    --out "${CALIB_JSONL}" --n 8000
fi

RAW="${SCORES_ROOT}/internvl_raw.pt"
ZSCORE="${SCORES_ROOT}/internvl_eic.pt"

python -m causal_core.calibrate \
  --model_name "${MODEL_INTERNVL}" \
  --model_type internvl \
  --question_file "${CALIB_JSONL}" \
  --image_folder "${COCO_DIR}/val2014" \
  --n_samples 8000 \
  --all_layers \
  --variance_mode env_per_example \
  --amp_dtype bf16 \
  --out "${RAW}"

python -m causal_core.apply_zscore_filter \
  --input "${RAW}" \
  --output "${ZSCORE}"

echo "[done] InternVL EIC -> ${ZSCORE}"
