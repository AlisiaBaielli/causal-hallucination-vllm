#!/usr/bin/env bash
# Reproduce all five decode-time baselines + Ours (chall) on LLaVA CHAIR.
# Thesis settings: seed 3407, n=500, max_new_tokens=128, layer=1, alpha=0.7.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
cd "${REPO_ROOT}"

SCORES="${SCORES:-${SCORES_ROOT}/llava_eic.pt}"
BASE="${OUT_ROOT}/reproduce/llava_chair"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
mkdir -p "${BASE}" "${OUT_ROOT}/slurm"

if [[ ! -f "${SCORES}" ]]; then
  echo "Missing calibration scores: ${SCORES}" >&2
  echo "Run: sbatch scripts/calibrate/llava.sh  (or symlink scores/llava_eic.pt)" >&2
  exit 1
fi

run_llava_chair_method() {
  local method="$1"
  shift
  local out="${BASE}/${method}"
  local cap="${out}/${method}.jsonl"
  local metrics="${out}/chair_results.json"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${metrics}" ]]; then
    echo "=== ${method} skipped (${metrics} exists) ==="
    return 0
  fi

  mkdir -p "${out}"
  echo "=== ${method} start ==="

  python experiments/chair/llava.py \
    --seed 3407 \
    --model_path "${MODEL_LLAVA}" \
    --data_path "${COCO_DIR}/val2014" \
    --anno_path "${COCO_DIR}/annotations/instances_val2014.json" \
    --out_path "${out}" \
    --num_eval_samples 500 \
    --max_new_tokens 128 \
    --method_name "${method}" \
    "$@"

  run_chair_metrics "${cap}" "${metrics}"
  echo "=== ${method} done ==="
}

run_llava_chair_method vanilla --no_hook
run_llava_chair_method vcd --no_hook --use_vcd
run_llava_chair_method m3id --no_hook --use_m3id
run_llava_chair_method only --no_hook --use_only --layer_index 1
run_llava_chair_method only_eic --no_hook --use_only --use_eic_heads \
  --c_scores_path "${SCORES}" --layer_index 1
run_llava_chair_method chall --c_scores_path "${SCORES}" --layer_index 1 --alpha 0.7

echo ""
echo "All methods under ${BASE}"
BASE="${BASE}" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["BASE"])
methods = ["vanilla", "vcd", "m3id", "only", "only_eic", "chall"]
print(f"{'Method':<10} {'CHAIRs':>8} {'CHAIRi':>8} {'Recall':>8}")
for m in methods:
    p = base / m / "chair_results.json"
    if not p.exists():
        print(f"{m:<10} {'—':>8} {'—':>8} {'—':>8}")
        continue
    om = json.loads(p.read_text()).get("overall_metrics", {})
    print(f"{m:<10} {om.get('CHAIRs', 0)*100:8.1f} {om.get('CHAIRi', 0)*100:8.1f} {om.get('Recall', 0)*100:8.1f}")
PY
