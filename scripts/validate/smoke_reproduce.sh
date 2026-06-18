#!/usr/bin/env bash
# Validate the full reproduce pipeline (scripts/reproduce/run.sh) end-to-end at
# n=1 for one model across ALL benchmarks. Exercises every orchestration branch:
# method loop, caption/answer globbing, CHAIR/POPE/AMBER/MME scoring (incl.
# eval/mme_score.py + AMBER toolkit), capability evals, and the summary tables.
# Proves "I could run anything" WITHOUT a full-scale run.
set -uo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_chair_deps

MODEL="${1:-llava}"
SM="${OUT_ROOT}/smoke_reproduce"
TMP="${SM}/_inputs"
mkdir -p "${TMP}/POPE"

# --- tiny input subsets ---
head -n 4 "${MME_QUESTIONS}" > "${TMP}/mme_smoke.jsonl"
head -n 4 "${POPE_DIR}/coco_pope_random.json" > "${TMP}/POPE/coco_pope_random.json"
AMBER_QUERY="${AMBER_QUERY}" TMP="${TMP}" python - <<'PY'
import json, os
src = os.environ["AMBER_QUERY"]; out = os.path.join(os.environ["TMP"], "amber_smoke.json")
q = json.loads(open(src).read())
json.dump(q[:1], open(out, "w"))
print("amber subset ->", out)
PY

# --- drive run.sh with overridden (tiny) inputs ---
export OUT_ROOT="${SM}"
export POPE_DIR="${TMP}/POPE"
export POPE_TYPES="random"
export MME_QUESTIONS="${TMP}/mme_smoke.jsonl"
export AMBER_QUERY="${TMP}/amber_smoke.json"
export NCHAIR=1
export CAP_LIMIT=1

BENCHES="chair pope amber mme"
[[ "${MODEL}" != "internvl" ]] && BENCHES="${BENCHES} mmvp mmbench"

PASS=0; FAIL=0
for b in ${BENCHES}; do
  echo ""; echo "########## reproduce ${MODEL} ${b} ##########"
  if bash scripts/reproduce/run.sh "${MODEL}" "${b}"; then
    echo "[PASS] reproduce ${MODEL} ${b}"; PASS=$((PASS + 1))
  else
    echo "[FAIL] reproduce ${MODEL} ${b}"; FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=========================================="
echo "REPRODUCE SMOKE (${MODEL}): ${PASS} passed, ${FAIL} failed"
echo "=========================================="
[[ "${FAIL}" -eq 0 ]]
