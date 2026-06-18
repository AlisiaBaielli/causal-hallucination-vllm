#!/usr/bin/env bash
# Broad GPU smoke: one cheap run per model / benchmark / baseline path (~1 h on A100).
# Does NOT check thesis numbers — only that each pipeline completes without error.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_transformers_v5
require_chair_deps

OUT="${OUT_ROOT}/smoke_all"
mkdir -p "${OUT}"
TMP="${OUT}/tmp"
mkdir -p "${TMP}"

PASS=0
FAIL=0
ok() { echo "[PASS] $1"; PASS=$((PASS + 1)); }
bad() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

run_step() {
  local name="$1"
  shift
  echo ""
  echo "=== ${name} ==="
  if "$@"; then
    ok "${name}"
  else
    bad "${name}"
  fi
}

# --- LLaVA CHAIR (vanilla + Ours) ---
run_step "LLaVA CHAIR" bash scripts/validate/smoke_chair.sh

# --- Qwen3 CHAIR (Ours + VCD baseline path) ---
run_step "Qwen3 CHAIR Ours" python experiments/chair/qwen3.py \
  --seed 3407 \
  --model_path "${MODEL_QWEN3}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/qwen3_chall" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" \
  --layer_index 0 \
  --alpha 0.7 \
  --method_name chall

run_step "Qwen3 CHAIR VCD" python experiments/chair/qwen3.py \
  --seed 3407 \
  --model_path "${MODEL_QWEN3}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/qwen3_vcd" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" \
  --layer_index 0 \
  --alpha 0.7 \
  --method_name vcd \
  --use_vcd

CAP="$(chall_caption_path "${OUT}/qwen3_chall" 0.7 chall)"
if [[ -s "${CAP}" ]]; then
  run_step "Qwen3 CHAIR score" run_chair_metrics "${CAP}" "${OUT}/qwen3_chall/chair.json"
else
  echo "[SKIP] Qwen3 CHAIR score (no captions at ${CAP})"
fi

# --- InternVL CHAIR (vanilla + Ours) ---
run_step "InternVL CHAIR vanilla" python experiments/chair/internvl.py \
  --seed 3407 \
  --model_path "${MODEL_INTERNVL}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/internvl_vanilla" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --c_scores_path "${SCORES_ROOT}/internvl_eic.pt" \
  --layer_index 0 \
  --alpha 0.7 \
  --method_name vanilla \
  --no_hook

run_step "InternVL CHAIR Ours" python experiments/chair/internvl.py \
  --seed 3407 \
  --model_path "${MODEL_INTERNVL}" \
  --data_path "${COCO_DIR}/val2014" \
  --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
  --out_path "${OUT}/internvl_chall" \
  --num_eval_samples 1 \
  --max_new_tokens 16 \
  --c_scores_path "${SCORES_ROOT}/internvl_eic.pt" \
  --layer_index 0 \
  --alpha 0.7 \
  --method_name chall

# --- LLaVA POPE (5 questions) ---
POPE_FILE="${POPE_DIR}/coco_pope_random.json"
if [[ -f "${POPE_FILE}" ]]; then
  head -5 "${POPE_FILE}" > "${TMP}/pope_smoke.json"
  run_step "LLaVA POPE" python experiments/pope/llava.py \
    --seed 42 \
    --model_path "${MODEL_LLAVA}" \
    --data_path "${COCO_DIR}/val2014" \
    --pope_path "${TMP}/pope_smoke.json" \
    --out_path "${OUT}/llava_pope" \
    --c_scores_path "${SCORES_ROOT}/llava_eic.pt" \
    --layer_index 1 \
    --alpha 0.7 \
    --type random \
    --dataset_name coco
else
  echo "[SKIP] LLaVA POPE (missing ${POPE_FILE})"
fi

# --- LLaVA AMBER (1 query) ---
if [[ -f "${AMBER_QUERY}" ]]; then
  python - <<PY
import json
from pathlib import Path
src = Path("${AMBER_QUERY}")
out = Path("${TMP}/amber_smoke.json")
queries = json.loads(src.read_text())
out.write_text(json.dumps(queries[:1]))
print(f"wrote 1 AMBER query -> {out}")
PY
  run_step "LLaVA AMBER gen" python experiments/amber/llava.py \
    --seed 42 \
    --model_path "${MODEL_LLAVA}" \
    --amber_query "${TMP}/amber_smoke.json" \
    --amber_image_dir "${AMBER_IMAGE_DIR}" \
    --output_file "${OUT}/llava_amber/amber_chall.json" \
    --c_scores_path "${SCORES_ROOT}/llava_eic.pt" \
    --layer_index 1 \
    --alpha 0.7

  if [[ -d "${AMBER_TOOLKIT}" ]]; then
    run_step "LLaVA AMBER score" bash -c \
      "cd '${AMBER_TOOLKIT}' && python inference.py --inference_data '${OUT}/llava_amber/amber_chall.json' --evaluation_type g"
  else
    echo "[SKIP] LLaVA AMBER score (no ${AMBER_TOOLKIT})"
  fi
else
  echo "[SKIP] LLaVA AMBER (missing ${AMBER_QUERY})"
fi

# --- LLaVA MME (1 question) ---
MME_QUESTIONS="${MME_DIR}/test_merged_final.jsonl"
if [[ -f "${MME_QUESTIONS}" ]]; then
  head -1 "${MME_QUESTIONS}" > "${TMP}/mme_smoke.jsonl"
  run_step "LLaVA MME" python experiments/mme/llava.py \
    --seed 42 \
    --model_path "${MODEL_LLAVA}" \
    --image_folder "${MME_IMAGE_DIR}" \
    --question_file "${TMP}/mme_smoke.jsonl" \
    --answers_file "${OUT}/llava_mme/mme_chall.jsonl" \
    --c_scores_path "${SCORES_ROOT}/llava_eic.pt" \
    --layer_index 1 \
    --alpha 0.7
else
  echo "[SKIP] LLaVA MME (missing ${MME_QUESTIONS})"
fi

echo ""
echo "=========================================="
echo "SMOKE ALL SUMMARY: ${PASS} passed, ${FAIL} failed"
echo "Results under ${OUT}"
echo "=========================================="
[[ "${FAIL}" -eq 0 ]]
