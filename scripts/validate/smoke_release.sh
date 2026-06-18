#!/usr/bin/env bash
# Full-coverage release smoke: every model x benchmark x method (+ LLaVA
# ablations + all mechanistic) at 1 sample / tiny n. Checks that each pipeline
# completes without error -- NOT thesis numbers.
#
# Select models via SMOKE_MODELS (default "llava qwen3 internvl").
set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_transformers_v5
require_chair_deps

MODELS="${SMOKE_MODELS:-llava qwen3 internvl}"
OUT="${OUT_ROOT}/smoke_release"
TMP="${OUT}/tmp"
mkdir -p "${OUT}" "${TMP}"

PASS=0; FAIL=0; SKIP=0
ok()   { echo "[PASS] $1"; PASS=$((PASS + 1)); }
bad()  { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "[SKIP] $1"; SKIP=$((SKIP + 1)); }
run_step() {
  local name="$1"; shift
  echo ""; echo "=== ${name} ==="
  if "$@"; then ok "${name}"; else bad "${name}"; fi
}

COCO_IMG="${COCO_DIR}/val2014"
COCO_ANNO="${COCO_DIR}/annotations/instances_val2014.json"
POPE_FILE="${POPE_DIR}/coco_pope_random.json"
[[ -f "${POPE_FILE}" ]] && head -5 "${POPE_FILE}" > "${TMP}/pope.json"
if [[ -f "${MME_QUESTIONS}" ]]; then head -1 "${MME_QUESTIONS}" > "${TMP}/mme.jsonl"; fi
if [[ -f "${AMBER_QUERY}" ]]; then
  python - "$AMBER_QUERY" "${TMP}/amber.json" <<'PY'
import json, sys
q = json.loads(open(sys.argv[1]).read())
open(sys.argv[2], "w").write(json.dumps(q[:1]))
PY
fi

# ---------------------------------------------------------------- LLaVA
smoke_llava() {
  local M="${MODEL_LLAVA}" S="${SCORES_ROOT}/llava_eic.pt" L=1
  local CH="experiments/chair/llava.py"
  run_step "LLaVA CHAIR vanilla" python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/chair_vanilla" --num_eval_samples 1 --max_new_tokens 16 --no_hook --method_name vanilla
  run_step "LLaVA CHAIR chall"   python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/chair_chall" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --method_name chall
  run_step "LLaVA CHAIR ONLY"    python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/chair_only" --num_eval_samples 1 --max_new_tokens 16 --use_only
  run_step "LLaVA CHAIR ONLY+EIC" python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/chair_only_eic" --num_eval_samples 1 --max_new_tokens 16 --use_only --use_eic_heads --c_scores_path "${S}" --layer_index ${L}
  run_step "LLaVA CHAIR VCD"     python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/chair_vcd" --num_eval_samples 1 --max_new_tokens 16 --use_vcd --method_name vcd
  run_step "LLaVA CHAIR M3ID"    python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/chair_m3id" --num_eval_samples 1 --max_new_tokens 16 --use_m3id --method_name m3id

  if [[ -f "${TMP}/pope.json" ]]; then
    run_step "LLaVA POPE chall" python experiments/pope/llava.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/llava/pope_chall" --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --type random --dataset_name coco
    run_step "LLaVA POPE ONLY"  python experiments/pope/llava.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/llava/pope_only" --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --type random --dataset_name coco --use_only
  else skip "LLaVA POPE (no ${POPE_FILE})"; fi

  if [[ -f "${TMP}/amber.json" ]]; then
    run_step "LLaVA AMBER chall" python experiments/amber/llava.py --model_path "${M}" --amber_query "${TMP}/amber.json" --amber_image_dir "${AMBER_IMAGE_DIR}" --output_file "${OUT}/llava/amber_chall/amber.json" --c_scores_path "${S}" --layer_index ${L} --alpha 0.7
    run_step "LLaVA AMBER ONLY"  python experiments/amber/llava.py --model_path "${M}" --amber_query "${TMP}/amber.json" --amber_image_dir "${AMBER_IMAGE_DIR}" --output_file "${OUT}/llava/amber_only/amber.json" --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --use_only
  else skip "LLaVA AMBER (no ${AMBER_QUERY})"; fi

  if [[ -f "${TMP}/mme.jsonl" ]]; then
    run_step "LLaVA MME chall" python experiments/mme/llava.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/llava/mme_chall/mme.jsonl" --c_scores_path "${S}" --layer_index ${L} --alpha 0.7
    run_step "LLaVA MME ONLY"  python experiments/mme/llava.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/llava/mme_only/mme.jsonl" --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --use_only
  else skip "LLaVA MME (no ${MME_QUESTIONS})"; fi

  # Ablations (LLaVA)
  run_step "ABL eic_suppress_chair" python experiments/ablations/eic_suppress_chair.py --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/abl_eic_suppress_chair" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L}
  run_step "ABL tver_head_select"   python experiments/ablations/tver_head_select_chair.py --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/abl_tver_head" --num_eval_samples 1 --max_new_tokens 16 --layer_index ${L} --alpha 0.7
  run_step "ABL tver_signal"        python experiments/ablations/tver_signal_chair.py --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/abl_tver_signal" --num_eval_samples 1 --max_new_tokens 16 --layer_index ${L} --alpha 0.7
  if [[ -f "${TMP}/pope.json" ]]; then
    run_step "ABL eic_suppress_pope" python experiments/ablations/eic_suppress_pope.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/llava/abl_eic_suppress_pope" --c_scores_path "${S}" --layer_index ${L} --type random --dataset_name coco
  fi
  if [[ -f "${TMP}/mme.jsonl" ]]; then
    run_step "ABL eic_suppress_mme" python experiments/ablations/eic_suppress_mme.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/llava/abl_eic_suppress_mme/mme.jsonl" --c_scores_path "${S}" --layer_index ${L}
  fi
  run_step "ABL only_eic_runner" python -m experiments.ablations.only_eic_runner chair --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/llava/abl_only_eic_runner" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L}

  # Mechanistic
  if [[ -f "${TMP_POPE_POP:=${POPE_DIR}/coco_pope_popular.json}" ]]; then
    run_step "LLaVA mechanistic" python experiments/head_influence_llava.py --model_path "${M}" --coco_dir "${COCO_IMG}" --pope_file "${POPE_DIR}/coco_pope_popular.json" --scores_path "${S}" --enhance_layer_index ${L} --n_generative 4 --n_yesno 4 --output_dir "${OUT}/llava/mech"
  else skip "LLaVA mechanistic (no popular POPE)"; fi
}

# ---------------------------------------------------------------- Qwen3
smoke_qwen3() {
  local M="${MODEL_QWEN3}" S="${SCORES_ROOT}/qwen3_eic.pt" L=0
  local CH="experiments/chair/qwen3.py"
  run_step "Qwen3 CHAIR vanilla" python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/qwen3/chair_vanilla" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --no_hook --method_name vanilla
  run_step "Qwen3 CHAIR chall"   python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/qwen3/chair_chall" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --method_name chall
  run_step "Qwen3 CHAIR ONLY"    python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/qwen3/chair_only" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --use_only
  run_step "Qwen3 CHAIR VCD"     python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/qwen3/chair_vcd" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --use_vcd --method_name vcd
  run_step "Qwen3 CHAIR M3ID"    python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/qwen3/chair_m3id" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --use_m3id --method_name m3id

  if [[ -f "${TMP}/pope.json" ]]; then
    run_step "Qwen3 POPE chall" python experiments/pope/qwen3.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/qwen3/pope_chall" --c_scores_path "${S}" --layer_index ${L} --alpha 0.3
    run_step "Qwen3 POPE ONLY"  python experiments/pope/qwen3.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/qwen3/pope_only" --c_scores_path "${S}" --layer_index ${L} --use_only
  else skip "Qwen3 POPE (no POPE)"; fi
  if [[ -f "${TMP}/amber.json" ]]; then
    run_step "Qwen3 AMBER chall" python experiments/amber/qwen3.py --model_path "${M}" --amber_query "${TMP}/amber.json" --amber_image_dir "${AMBER_IMAGE_DIR}" --output_file "${OUT}/qwen3/amber_chall/amber.json" --c_scores_path "${S}" --layer_index ${L} --alpha 0.3
    run_step "Qwen3 AMBER ONLY"  python experiments/amber/qwen3.py --model_path "${M}" --amber_query "${TMP}/amber.json" --amber_image_dir "${AMBER_IMAGE_DIR}" --output_file "${OUT}/qwen3/amber_only/amber.json" --c_scores_path "${S}" --layer_index ${L} --use_only
  else skip "Qwen3 AMBER (no AMBER)"; fi
  if [[ -f "${TMP}/mme.jsonl" ]]; then
    run_step "Qwen3 MME chall" python experiments/mme/qwen3.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/qwen3/mme_chall/mme.jsonl" --c_scores_path "${S}" --layer_index ${L} --alpha 0.3
    run_step "Qwen3 MME ONLY"  python experiments/mme/qwen3.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/qwen3/mme_only/mme.jsonl" --c_scores_path "${S}" --layer_index ${L} --use_only
  else skip "Qwen3 MME (no MME)"; fi

  if [[ -f "${POPE_DIR}/coco_pope_popular.json" ]]; then
    run_step "Qwen3 mechanistic" python experiments/head_influence_qwen3.py --model_path "${M}" --coco_dir "${COCO_IMG}" --pope_file "${POPE_DIR}/coco_pope_popular.json" --scores_path "${S}" --enhance_layer_index ${L} --n_generative 4 --n_yesno 4 --output_dir "${OUT}/qwen3/mech"
  fi
}

# ---------------------------------------------------------------- InternVL
smoke_internvl() {
  local M="${MODEL_INTERNVL}" S="${SCORES_ROOT}/internvl_eic.pt" L=0
  local CH="experiments/chair/internvl.py"
  run_step "InternVL CHAIR vanilla" python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/internvl/chair_vanilla" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --no_hook --method_name vanilla
  run_step "InternVL CHAIR chall"   python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/internvl/chair_chall" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --method_name chall
  run_step "InternVL CHAIR ONLY"    python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/internvl/chair_only" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --use_only --no_hook --method_name only
  run_step "InternVL CHAIR VCD"     python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/internvl/chair_vcd" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --use_vcd --method_name vcd
  run_step "InternVL CHAIR M3ID"    python "${CH}" --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" --out_path "${OUT}/internvl/chair_m3id" --num_eval_samples 1 --max_new_tokens 16 --c_scores_path "${S}" --layer_index ${L} --use_m3id --method_name m3id

  if [[ -f "${TMP}/pope.json" ]]; then
    run_step "InternVL POPE chall" python experiments/pope/internvl.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/internvl/pope_chall" --c_scores_path "${S}" --layer_index ${L} --alpha 0.3
    run_step "InternVL POPE ONLY"  python experiments/pope/internvl.py --model_path "${M}" --data_path "${COCO_IMG}" --pope_path "${TMP}/pope.json" --out_path "${OUT}/internvl/pope_only" --c_scores_path "${S}" --layer_index ${L} --use_only --no_hook
  else skip "InternVL POPE (no POPE)"; fi
  if [[ -f "${TMP}/amber.json" ]]; then
    run_step "InternVL AMBER chall" python experiments/amber/internvl.py --model_path "${M}" --amber_query "${TMP}/amber.json" --amber_image_dir "${AMBER_IMAGE_DIR}" --output_file "${OUT}/internvl/amber_chall/amber.json" --c_scores_path "${S}" --layer_index ${L} --alpha 0.3
    run_step "InternVL AMBER ONLY"  python experiments/amber/internvl.py --model_path "${M}" --amber_query "${TMP}/amber.json" --amber_image_dir "${AMBER_IMAGE_DIR}" --output_file "${OUT}/internvl/amber_only/amber.json" --c_scores_path "${S}" --layer_index ${L} --use_only --no_hook
  else skip "InternVL AMBER (no AMBER)"; fi
  if [[ -f "${TMP}/mme.jsonl" ]]; then
    run_step "InternVL MME chall" python experiments/mme/internvl.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/internvl/mme_chall/mme.jsonl" --c_scores_path "${S}" --layer_index ${L} --alpha 0.3
    run_step "InternVL MME ONLY"  python experiments/mme/internvl.py --model_path "${M}" --image_folder "${MME_IMAGE_DIR}" --question_file "${TMP}/mme.jsonl" --answers_file "${OUT}/internvl/mme_only/mme.jsonl" --c_scores_path "${S}" --layer_index ${L} --use_only --no_hook
  else skip "InternVL MME (no MME)"; fi

  if [[ -f "${POPE_DIR}/coco_pope_popular.json" ]]; then
    run_step "InternVL mechanistic" python experiments/head_influence_internvl.py --model_path "${M}" --coco_dir "${COCO_IMG}" --pope_file "${POPE_DIR}/coco_pope_popular.json" --scores_path "${S}" --enhance_layer_index ${L} --n_generative 4 --n_yesno 4 --output_dir "${OUT}/internvl/mech"
  fi
}

for m in ${MODELS}; do
  case "${m}" in
    llava)    smoke_llava ;;
    qwen3)    smoke_qwen3 ;;
    internvl) smoke_internvl ;;
    *) echo "unknown model: ${m}" ;;
  esac
done

echo ""
echo "=========================================="
echo "RELEASE SMOKE [${MODELS}]: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "Results under ${OUT}"
echo "=========================================="
[[ "${FAIL}" -eq 0 ]]
