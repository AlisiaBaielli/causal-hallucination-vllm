#!/bin/bash
# EIC head suppression vs Ours sharpening ablation.
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
LAYER="${LAYER:-1}"
NCHAIR="${NCHAIR:-500}"
POPE_TYPE="${POPE_TYPE:-random}"
OUT="${OUT_ROOT}/ablation_eic_suppress"
mkdir -p "${OUT}/chair" "${OUT}/pope" "${OUT}/mme"

# CHAIR (long-form): writes ${OUT}/chair/eic_suppress.jsonl
python experiments/ablations/eic_suppress_chair.py \
  --model_path "${MODEL_LLAVA}" --c_scores_path "${SCORES}" --layer_index "${LAYER}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --num_eval_samples "${NCHAIR}" --out_path "${OUT}/chair"
run_chair_metrics "${OUT}/chair/eic_suppress.jsonl" "${OUT}/chair_results.json"

# POPE (random split)
python experiments/ablations/eic_suppress_pope.py \
  --model_path "${MODEL_LLAVA}" --c_scores_path "${SCORES}" --layer_index "${LAYER}" \
  --data_path "${COCO_DIR}/val2014" \
  --pope_path "${POPE_DIR}/coco_pope_${POPE_TYPE}.json" \
  --type "${POPE_TYPE}" --dataset_name coco --out_path "${OUT}/pope"

# MME (scored with the in-repo scorer)
python experiments/ablations/eic_suppress_mme.py \
  --model_path "${MODEL_LLAVA}" --c_scores_path "${SCORES}" --layer_index "${LAYER}" \
  --image_folder "${MME_IMAGE_DIR}" --question_file "${MME_QUESTIONS}" \
  --answers_file "${OUT}/mme/eic_suppress.jsonl"
python eval/mme_score.py --answers_file "${OUT}/mme/eic_suppress.jsonl" \
  --question_file "${MME_QUESTIONS}" --out_path "${OUT}/mme/mme_score.json"

echo "=== EIC suppress ablation done ==="
