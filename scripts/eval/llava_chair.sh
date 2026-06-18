

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
OUT="${OUT_ROOT}/llava_chair_chall_a0.7"
mkdir -p "${OUT}"

python experiments/chair/llava.py \
    --seed 3407 \
    --model_path "${MODEL_LLAVA}" \
    --data_path "${COCO_DIR}/val2014" \
    --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
    --out_path "${OUT}" \
    --c_scores_path "${SCORES}" \
    --layer_index 1 \
    --alpha 0.7 \
    --num_eval_samples 500 \
    --max_new_tokens 128 \
    --method_name chall

run_chair_metrics "${OUT}/chall.jsonl" "${OUT}/chair_results.json"
echo "=== LLaVA CHAIR CHALL alpha=0.7 ==="
