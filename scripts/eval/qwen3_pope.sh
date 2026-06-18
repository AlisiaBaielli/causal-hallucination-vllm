

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/qwen3_eic.pt}"
OUT="${OUT_ROOT}/qwen3_pope_chall"
mkdir -p "${OUT}"

python experiments/pope/qwen3.py \
    --seed 42 \
    --model_path "${MODEL_QWEN3}" \
    --data_path "${COCO_DIR}/val2014" \
    --pope_path "${POPE_DIR}/coco_pope_random.json" \
    --c_scores_path "${SCORES}" \
    --layer_index 0 \
    --alpha 0.7 \
    --out_path "${OUT}"

echo "=== Qwen3 POPE CHALL done ==="
