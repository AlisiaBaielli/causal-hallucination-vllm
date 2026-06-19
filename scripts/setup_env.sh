#!/usr/bin/env bash
# Create or update the unified conda env for all models (LLaVA, Qwen3-VL, InternVL).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-chall}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required. Load Anaconda module first on the cluster." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Updating existing env: ${ENV_NAME}"
  conda activate "${ENV_NAME}"
else
  echo "Creating env: ${ENV_NAME} (python=3.10)"
  conda create -n "${ENV_NAME}" python=3.10 -y
  conda activate "${ENV_NAME}"
fi

python -m pip install -U pip wheel
python -m pip install -r "${REPO_ROOT}/requirements.txt"

export REPO_ROOT
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/transformers/src"
export PYTHONSTARTUP="${REPO_ROOT}/causal_core/_python_startup.py"

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "  export PYTHONPATH=\"\$(pwd):\$(pwd)/transformers/src:\${PYTHONPATH:-}\""
echo "  export PYTHONSTARTUP=\"\$(pwd)/causal_core/_python_startup.py\""
echo "Or source scripts/_env.sh from any run script."
