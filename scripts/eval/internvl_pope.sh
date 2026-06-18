

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/internvl_eic.pt}"
OUT="${OUT_ROOT}/internvl_pope_chall"
mkdir -p "${OUT}"

python experiments/pope/internvl.py \
    --seed 42 \
    --model_path "${MODEL_INTERNVL}" \
    --data_path "${COCO_DIR}/val2014" \
    --pope_path "${POPE_DIR}/coco_pope_random.json" \
    --c_scores_path "${SCORES}" \
    --layer_index 1 \
    --alpha 0.7 \
    --out_path "${OUT}"

echo "=== InternVL POPE CHALL done ==="
