#!/usr/bin/env bash
# Focused GPU smoke for the LLaVA ONLY contrastive-decoding branch (TF5 fork).
# Verifies: ONLY baseline, ONLY+EIC, the contrastive mechanistic (~rho>0),
# and that the CHALL monitor path still runs after the fork changes.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_transformers_v5
require_chair_deps

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
if [ ! -f "${SCORES}" ]; then
  echo "Missing ${SCORES}. Run: bash scripts/calibrate/llava.sh"
  exit 1
fi

OUT="${OUT_ROOT}/smoke_llava_only"
mkdir -p "${OUT}"

PASS=0
FAIL=0
ok() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
bad() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
run_step() {
  local name="$1"; shift
  echo ""; echo "=== ${name} ==="
  if "$@"; then ok "${name}"; else bad "${name}"; fi
}

# --- LLaVA ONLY baseline (CD branch must run) ---
run_step "LLaVA CHAIR ONLY" python experiments/chair/llava.py \
  --seed 3407 \
  --model_path "${MODEL_LLAVA}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/only" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --use_only

# --- LLaVA ONLY+EIC (CD branch + EIC head set) ---
run_step "LLaVA CHAIR ONLY+EIC" python experiments/chair/llava.py \
  --seed 3407 \
  --model_path "${MODEL_LLAVA}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/only_eic" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --use_only --use_eic_heads \
  --c_scores_path "${SCORES}" \
  --layer_index 1

# --- CHALL monitor path regression (must still work) ---
run_step "LLaVA CHAIR CHALL" python experiments/chair/llava.py \
  --seed 3407 \
  --model_path "${MODEL_LLAVA}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/chall" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --c_scores_path "${SCORES}" \
  --layer_index 1 \
  --alpha 0.7 \
  --method_name chall

# --- Contrastive mechanistic (small n; should give positive correlation) ---
POPE_FILE="${POPE_DIR}/coco_pope_popular.json"
if [[ -f "${POPE_FILE}" ]]; then
  run_step "LLaVA mechanistic (contrastive)" python experiments/head_influence_llava.py \
    --model_path "${MODEL_LLAVA}" \
    --coco_dir "${COCO_DIR}/val2014" \
    --pope_file "${POPE_FILE}" \
    --scores_path "${SCORES}" \
    --enhance_layer_index 1 \
    --n_generative 12 \
    --n_yesno 12 \
    --output_dir "${OUT}/mech"
else
  echo "[SKIP] LLaVA mechanistic (missing ${POPE_FILE})"
fi

echo ""
echo "=========================================="
echo "LLaVA ONLY SMOKE: ${PASS} passed, ${FAIL} failed"
echo "Results under ${OUT}"
echo "=========================================="
[[ "${FAIL}" -eq 0 ]]
