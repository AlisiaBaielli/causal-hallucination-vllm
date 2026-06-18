
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_transformers_v5
require_chair_deps

SCORES="${SCORES:-${SCORES_ROOT}/qwen3_eic.pt}"
ALPHA=0.7
OUT="${OUT_ROOT}/qwen3_chair_chall"
mkdir -p "${OUT}"

python experiments/chair/qwen3.py \
    --seed 3407 \
    --model_path "${MODEL_QWEN3}" \
    --data_path "${COCO_DIR}/val2014" \
    --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
    --out_path "${OUT}" \
    --c_scores_path "${SCORES}" \
    --layer_index 0 \
    --alpha "${ALPHA}" \
    --num_eval_samples 500 \
    --method_name chall

CAP="$(chall_caption_path "${OUT}" "${ALPHA}" chall)"
run_chair_metrics "${CAP}" "${OUT}/chair_results.json"
echo "=== Qwen3 CHAIR CHALL alpha=${ALPHA} ==="
