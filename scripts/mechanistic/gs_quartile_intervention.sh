
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

LAYER="${SLURM_ARRAY_TASK_ID:-1}"
SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
OUT="${OUT_ROOT}/mechanistic/gs_quartile_layer${LAYER}"
mkdir -p "${OUT}"

echo "=== GS quartile analysis: layer ${LAYER}, 2000 images ==="

python experiments/perhead_entropy.py \
  --seed 3407 \
  --model_path "${MODEL_LLAVA}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --c_scores_path "${SCORES}" \
  --out_path "${OUT}" \
  --num_eval_samples 2000 \
  --max_new_tokens 128 \
  --layer_index "${LAYER}" \
  --img_start 35 \
  --img_len 576

echo "[done] layer ${LAYER}"
