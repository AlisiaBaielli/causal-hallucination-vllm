
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

python -m experiments.ablations.make_head_variants \
  --input "${SCORES_ROOT}/llava_eic.pt" \
  --layer 1 \
  --out_dir "${SCORES_ROOT}" \
  --prefix llava_L1

case $SLURM_ARRAY_TASK_ID in
0)  TAG=random;   SCORES="${SCORES_ROOT}/llava_L1_random.pt" ;;
1)  TAG=all;      SCORES="${SCORES_ROOT}/llava_L1_all.pt" ;;
2)  TAG=inverted; SCORES="${SCORES_ROOT}/llava_L1_inverted.pt" ;;
esac

OUT="${OUT_ROOT}/head_ablation/llava_${TAG}"
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
    --num_eval_samples "${NSAMP:-500}" \
    --method_name "chall_${TAG}"

run_chair_metrics "${OUT}/chall_${TAG}.jsonl" "${OUT}/chair_results.json"
echo "=== LLaVA head ablation ${TAG} ==="
