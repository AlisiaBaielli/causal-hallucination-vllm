
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
OUT="${OUT_ROOT}/llava_amber_chall"
mkdir -p "${OUT}"

python experiments/amber/llava.py \
    --seed 42 \
    --model_path "${MODEL_LLAVA}" \
    --amber_query "${AMBER_QUERY}" \
    --amber_image_dir "${AMBER_IMAGE_DIR}" \
    --output_file "${OUT}/amber_chall.json" \
    --c_scores_path "${SCORES}" \
    --layer_index 1 \
    --alpha 0.7

cd "${AMBER_TOOLKIT}"
python inference.py \
    --inference_data "${OUT}/amber_chall.json" \
    --evaluation_type g

echo "=== LLaVA AMBER CHALL done ==="
