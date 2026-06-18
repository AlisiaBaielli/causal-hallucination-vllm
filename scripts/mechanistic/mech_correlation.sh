

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

OUT_BASE="${OUT_ROOT}/mechanistic/K7"
mkdir -p "${OUT_BASE}"

case $SLURM_ARRAY_TASK_ID in
0)
  SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
  OUT="${OUT_BASE}/llava"
  python experiments/head_influence_llava.py \
    --model_path "${MODEL_LLAVA}" \
    --coco_dir "${COCO_DIR}/val2014" \
    --pope_file "${POPE_DIR}/coco_pope_popular.json" \
    --scores_path "${SCORES}" \
    --enhance_layer_index 1 \
    --n_generative 200 --n_yesno 200 \
    --output_dir "${OUT}"
  ;;
1)
  SCORES="${SCORES:-${SCORES_ROOT}/qwen3_eic.pt}"
  OUT="${OUT_BASE}/qwen3"
  python experiments/head_influence_qwen3.py \
    --model_path "${MODEL_QWEN3}" \
    --coco_dir "${COCO_DIR}/val2014" \
    --pope_file "${POPE_DIR}/coco_pope_popular.json" \
    --scores_path "${SCORES}" \
    --enhance_layer_index 0 \
    --n_generative 200 --n_yesno 200 \
    --output_dir "${OUT}"
  ;;
2)
  SCORES="${SCORES:-${SCORES_ROOT}/internvl_eic.pt}"
  OUT="${OUT_BASE}/internvl"
  python experiments/head_influence_internvl.py \
    --model_path "${MODEL_INTERNVL}" \
    --coco_dir "${COCO_DIR}/val2014" \
    --pope_file "${POPE_DIR}/coco_pope_popular.json" \
    --scores_path "${SCORES}" \
    --enhance_layer_index 1 \
    --n_generative 200 --n_yesno 200 \
    --output_dir "${OUT}"
  ;;
esac

echo "[done] mechanistic task ${SLURM_ARRAY_TASK_ID}"
