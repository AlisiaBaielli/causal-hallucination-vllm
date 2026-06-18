#!/usr/bin/env bash
# Quick validation: imports, CLI help, and optional 1-image smoke test.
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
cd "${REPO_ROOT}"

echo "=== Python syntax check ==="
find causal_core experiments eval llava -name '*.py' -print0 | \
  xargs -0 python -m py_compile
echo "[ok] py_compile"

echo "=== Shell scripts (bash -n) ==="
find scripts -name '*.sh' -print0 | while IFS= read -r -d '' f; do
  bash -n "$f"
done
echo "[ok] bash syntax"

echo "=== Unified env verification ==="
bash scripts/validate/verify_env.sh

echo ""
echo "All static checks passed."
echo "For a GPU smoke test (1 image), run:"
echo "  bash scripts/validate/smoke_chair.sh"
