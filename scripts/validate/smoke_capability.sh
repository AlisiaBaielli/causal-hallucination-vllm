#!/usr/bin/env bash
# Capability-benchmark coverage smoke at tiny scale: confirms MMVP and MMBench
# evals execute end-to-end for BOTH LLaVA and Qwen3 across the thesis methods
# (vanilla / CHALL / ONLY / VCD). NOT thesis numbers.
#
# Requires: data/mmbench/mmbench_dev_20230712.tsv and MMVP cached in the HF hub
# cache (run once on a login node with network: see scripts/validate README).
set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

OUT="${OUT_ROOT}/smoke_capability"
mkdir -p "${OUT}"

MMBENCH_TSV="${MMBENCH_TSV:-${REPO_ROOT}/data/mmbench/mmbench_dev_20230712.tsv}"

PASS=0; FAIL=0; SKIP=0
ok()   { echo "[PASS] $1"; PASS=$((PASS + 1)); }
bad()  { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "[SKIP] $1"; SKIP=$((SKIP + 1)); }
run_step() { local n="$1"; shift; echo ""; echo "=== ${n} ==="; if "$@"; then ok "${n}"; else bad "${n}"; fi; }

# ---------------- MMVP (auto-downloads / HF cache) ----------------
# LLaVA
run_step "mmvp_llava_chall" python experiments/analysis/mmvp_eval.py \
  --model_path "${MODEL_LLAVA}" --c_scores_path "${SCORES_ROOT}/llava_eic.pt" \
  --layer_index 1 --alpha 0.7 --method_name chall --limit 2 \
  --out_path "${OUT}/mmvp_llava_chall"
run_step "mmvp_llava_only" python experiments/analysis/mmvp_eval.py \
  --model_path "${MODEL_LLAVA}" --c_scores_path "${SCORES_ROOT}/llava_eic.pt" \
  --layer_index 1 --use_only --method_name only --limit 2 \
  --out_path "${OUT}/mmvp_llava_only"

# Qwen3
run_step "mmvp_qwen3_chall" python experiments/analysis/mmvp_qwen3.py \
  --model_path "${MODEL_QWEN3}" --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" \
  --layer_index 0 --alpha 0.3 --method_name chall --limit 2 \
  --out_path "${OUT}/mmvp_qwen3_chall"
run_step "mmvp_qwen3_only" python experiments/analysis/mmvp_qwen3.py \
  --model_path "${MODEL_QWEN3}" --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" \
  --layer_index 0 --use_only --limit 2 \
  --out_path "${OUT}/mmvp_qwen3_only"
run_step "mmvp_qwen3_vcd" python experiments/analysis/mmvp_qwen3.py \
  --model_path "${MODEL_QWEN3}" --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" \
  --layer_index 0 --use_vcd --method_name vcd --limit 2 \
  --out_path "${OUT}/mmvp_qwen3_vcd"

# ---------------- MMBench (needs staged tsv) ----------------
if [[ -f "${MMBENCH_TSV}" ]]; then
  run_step "mmbench_llava_chall" python experiments/analysis/mmbench_eval.py \
    --model_path "${MODEL_LLAVA}" --question_file "${MMBENCH_TSV}" \
    --c_scores_path "${SCORES_ROOT}/llava_eic.pt" --layer_index 1 --alpha 0.3 \
    --method chall --limit 1 --out_dir "${OUT}/mmbench_llava_chall"
  run_step "mmbench_llava_only" python experiments/analysis/mmbench_eval.py \
    --model_path "${MODEL_LLAVA}" --question_file "${MMBENCH_TSV}" \
    --c_scores_path "${SCORES_ROOT}/llava_eic.pt" --layer_index 1 \
    --method only --limit 1 --out_dir "${OUT}/mmbench_llava_only"

  run_step "mmbench_qwen3_chall" python experiments/analysis/mmbench_qwen3.py \
    --model_path "${MODEL_QWEN3}" --question_file "${MMBENCH_TSV}" \
    --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" --layer_index 0 --alpha 0.3 \
    --method chall --limit 1 --out_dir "${OUT}/mmbench_qwen3_chall"
  run_step "mmbench_qwen3_only" python experiments/analysis/mmbench_qwen3.py \
    --model_path "${MODEL_QWEN3}" --question_file "${MMBENCH_TSV}" \
    --c_scores_path "${SCORES_ROOT}/qwen3_eic.pt" --layer_index 0 \
    --method only --limit 1 --out_dir "${OUT}/mmbench_qwen3_only"
else
  skip "mmbench_* (missing ${MMBENCH_TSV})"
fi

echo ""
echo "=========================================="
echo "CAPABILITY SMOKE: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "=========================================="
[[ "${FAIL}" -eq 0 ]]
