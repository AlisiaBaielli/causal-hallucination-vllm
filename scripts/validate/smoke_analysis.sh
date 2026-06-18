#!/usr/bin/env bash
# Analysis/mechanistic/ablation coverage smoke at tiny scale: confirms the
# thesis "Impact on Quality", efficiency, statistical, multilayer, per-token and
# per-head scripts all execute end-to-end. NOT thesis numbers.
#
# NOTE: MMVP / MMBench capability evals (LLaVA + Qwen3) are exercised separately
# by scripts/validate/smoke_capability.sh.
set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_chair_deps

OUT="${OUT_ROOT}/smoke_analysis"
TMP="${OUT}/tmp"
mkdir -p "${OUT}" "${TMP}"

M="${MODEL_LLAVA}"; S="${SCORES_ROOT}/llava_eic.pt"; L=1
COCO_IMG="${COCO_DIR}/val2014"
COCO_ANNO="${COCO_DIR}/annotations/instances_val2014.json"
COCO_CAPS="${COCO_DIR}/annotations/captions_val2014.json"

PASS=0; FAIL=0; SKIP=0
ok()   { echo "[PASS] $1"; PASS=$((PASS + 1)); }
bad()  { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "[SKIP] $1"; SKIP=$((SKIP + 1)); }
run_step() { local n="$1"; shift; echo ""; echo "=== ${n} ==="; if "$@"; then ok "${n}"; else bad "${n}"; fi; }

# ---- LLaVA generation analysis / mechanistic (TF5-risk paths) ----
run_step "efficiency_benchmark" python experiments/analysis/efficiency_benchmark.py \
  --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
  --c_scores_path "${S}" --layer_index ${L} --num_eval_samples 1 --warmup 0 \
  --max_new_tokens 16 --out_path "${OUT}/efficiency.json"

run_step "multilayer_chair" python experiments/analysis/multilayer_chair.py \
  --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
  --c_scores_paths "${S}" --layer_indices ${L} --num_eval_samples 1 \
  --max_new_tokens 16 --out_path "${OUT}/multilayer_chair"

run_step "multilayer_gs" python experiments/analysis/multilayer_gs.py \
  --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
  --c_scores_path "${S}" --layers "1" --num_eval_samples 1 \
  --max_new_tokens 16 --out_path "${OUT}/multilayer_gs"

run_step "perhead_entropy" python experiments/perhead_entropy.py \
  --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
  --c_scores_path "${S}" --layer_index ${L} --num_eval_samples 1 \
  --max_new_tokens 16 --out_path "${OUT}/perhead_entropy"

run_step "pertoken_analysis" python experiments/pertoken_analysis.py \
  --model_path "${M}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
  --c_scores_path "${S}" --layer_index ${L} --alpha 0.7 --num_eval_samples 1 \
  --max_new_tokens 16 --out_path "${OUT}/pertoken"

run_step "make_head_variants" python -m experiments.ablations.make_head_variants \
  --input "${S}" --layer ${L} --out_dir "${TMP}/variants" --prefix llava_L1

# ---- InternVL per-token ----
run_step "pertoken_analysis_internvl" python experiments/pertoken_analysis_internvl.py \
  --model_path "${MODEL_INTERNVL}" --data_path "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
  --c_scores_path "${SCORES_ROOT}/internvl_eic.pt" --layer_index 0 --num_eval_samples 1 \
  --max_new_tokens 16 --out_path "${OUT}/pertoken_internvl"

# ---- Post-hoc metrics on the smoke_release LLaVA captions ----
VAN="${OUT_ROOT}/smoke_release/llava/chair_vanilla/vanilla.jsonl"
CHALL="${OUT_ROOT}/smoke_release/llava/chair_chall/chall.jsonl"
if [[ -f "${VAN}" && -f "${CHALL}" ]]; then
  run_step "compute_meteor" python experiments/analysis/compute_meteor.py \
    --caption_files "vanilla:${VAN}" "chall:${CHALL}" --coco_captions "${COCO_CAPS}" \
    --out_path "${OUT}/meteor.json"
  run_step "compute_clipscore" python experiments/analysis/compute_clipscore.py \
    --caption_files "vanilla:${VAN}" --image_dir "${COCO_IMG}" --anno_path "${COCO_ANNO}" \
    --format jsonl --out_path "${OUT}/clipscore.json"
  run_chair_metrics "${VAN}" "${OUT}/van_chair.json" || true
  run_chair_metrics "${CHALL}" "${OUT}/chall_chair.json" || true
  if [[ -f "${OUT}/van_chair.json" && -f "${OUT}/chall_chair.json" ]]; then
    run_step "bootstrap_ci" python experiments/analysis/bootstrap_ci.py \
      --vanilla_results "${OUT}/van_chair.json" --chall_results "${OUT}/chall_chair.json"
  else skip "bootstrap_ci (no chair result jsons)"; fi
else
  skip "post-hoc metrics (run smoke_release llava first)"
fi

echo ""
echo "=========================================="
echo "ANALYSIS SMOKE: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "=========================================="
[[ "${FAIL}" -eq 0 ]]
