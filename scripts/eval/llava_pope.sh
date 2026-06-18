

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
OUT="${OUT_ROOT}/llava_pope_chall"
mkdir -p "${OUT}"

for TYPE in random popular adversarial; do
    python experiments/pope/llava.py \
        --seed 42 \
        --model_path "${MODEL_LLAVA}" \
        --data_path "${COCO_DIR}/val2014" \
        --pope_path "${POPE_DIR}/coco_pope_${TYPE}.json" \
        --c_scores_path "${SCORES}" \
        --layer_index 1 \
        --alpha 0.7 \
        --type "${TYPE}" \
        --dataset_name coco \
        --out_path "${OUT}"
    echo "=== LLaVA POPE ${TYPE} done ==="
done
