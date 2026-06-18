

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/qwen3_eic.pt}"
OUT="${OUT_ROOT}/qwen3_mme_chall"
mkdir -p "${OUT}"

python experiments/mme/qwen3.py \
    --seed 42 \
    --model_path "${MODEL_QWEN3}" \
    --image_folder "${MME_IMAGE_DIR}" \
    --question_file "${MME_QUESTIONS}" \
    --answers_file "${OUT}/mme_chall.jsonl" \
    --c_scores_path "${SCORES}" \
    --layer_index 0 \
    --alpha 0.7

echo "=== Qwen3 MME CHALL done ==="
