#!/usr/bin/env bash
# Verify the unified chall env: imports, deps, and eval CLIs for all three models.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi

setup_env
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV:-chall}"; then
    conda activate "${CONDA_ENV:-chall}"
  fi
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/transformers/src:${PYTHONPATH:-}"
export PYTHONSTARTUP="${REPO_ROOT}/causal_core/_python_startup.py"

PASS=0
FAIL=0
ok() { echo "[OK] $1"; PASS=$((PASS + 1)); }
bad() { echo "[FAIL] $1"; FAIL=$((FAIL + 1)); }

echo "=== Python: $(python -V) ==="
python - <<'PY' && ok "transformers>=5" || bad "transformers>=5"
import transformers
from packaging import version
assert version.parse(transformers.__version__) >= version.parse("5.0.0"), transformers.__version__
print(f"transformers {transformers.__version__}")
PY

require_chair_deps && ok "CHAIR deps (nltk, pycocotools)" || bad "CHAIR deps"

python - <<'PY' && ok "transformers fork (qwen3_vl, internvl)" || bad "transformers fork"
from causal_core.transformers_fork import ensure_all_forks
ensure_all_forks()
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.internvl.modeling_internvl_real import InternVLForConditionalGeneration
PY

python - <<'PY' && ok "causal_core imports" || bad "causal_core imports"
import causal_core.eval_common, causal_core.monitor, causal_core.scores, causal_core.only_eic
from causal_core.models.llava_sampling import evolve_only_sampling
PY

for f in \
  "${SCORES_ROOT}/llava_eic.pt" \
  "${SCORES_ROOT}/qwen3_eic.pt" \
  "${SCORES_ROOT}/internvl_eic.pt"; do
  [ -e "$f" ] && ok "score: $f" || bad "missing score: $f"
done

for py in \
  experiments/chair/llava.py \
  experiments/chair/qwen3.py \
  experiments/chair/internvl.py \
  experiments/pope/llava.py \
  experiments/pope/qwen3.py \
  experiments/pope/internvl.py \
  experiments/amber/llava.py \
  experiments/amber/qwen3.py \
  experiments/amber/internvl.py \
  experiments/mme/llava.py \
  experiments/mme/qwen3.py \
  experiments/mme/internvl.py; do
  python "${py}" --help >/dev/null && ok "${py}" || bad "${py}"
done

python -m causal_core.calibrate --help >/dev/null && ok "causal_core.calibrate" || bad "calibrate"

CAP="${REPO_ROOT}/results/qwen3_chair_chall/chall_alpha0.7_chall.jsonl"
if [ -f "${CAP}" ]; then
  head -1 "${CAP}" > /tmp/chall_verify_cap.jsonl
  run_chair_metrics /tmp/chall_verify_cap.jsonl /tmp/chall_verify_chair.json && ok "CHAIR scorer" || bad "CHAIR scorer"
else
  echo "[SKIP] CHAIR scorer (no sample captions at ${CAP})"
fi

echo ""
echo "SUMMARY: ${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ]
