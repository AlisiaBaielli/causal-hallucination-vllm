

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/internvl_eic.pt}"
OUT="${OUT_ROOT}/internvl_mme_chall"
mkdir -p "${OUT}"

python experiments/mme/internvl.py \
    --seed 42 \
    --model_path "${MODEL_INTERNVL}" \
    --image_folder "${MME_IMAGE_DIR}" \
    --question_file "${MME_QUESTIONS}" \
    --answers_file "${OUT}/mme_chall.jsonl" \
    --c_scores_path "${SCORES}" \
    --layer_index 1 \
    --alpha 0.7

echo "=== InternVL MME CHALL done ==="
