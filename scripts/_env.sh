
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results}"
SCORES_ROOT="${SCORES_ROOT:-${REPO_ROOT}/scores}"
CONDA_ENV="${CONDA_ENV:-chall}"

MODEL_LLAVA="${MODEL_LLAVA:-${REPO_ROOT}/data/models/llava-v1.5-7b}"
MODEL_QWEN3="${MODEL_QWEN3:-${REPO_ROOT}/data/models/Qwen3-VL-8B-Instruct}"
MODEL_INTERNVL="${MODEL_INTERNVL:-${REPO_ROOT}/data/models/InternVL3_5-8B-HF}"

COCO_DIR="${COCO_DIR:-${REPO_ROOT}/data/coco}"
POPE_DIR="${POPE_DIR:-${REPO_ROOT}/data/POPE/coco}"
AMBER_DIR="${AMBER_DIR:-${REPO_ROOT}/data/AMBER}"
AMBER_QUERY="${AMBER_QUERY:-${AMBER_DIR}/AMBER/data/query/query_generative.json}"
AMBER_IMAGE_DIR="${AMBER_IMAGE_DIR:-${AMBER_DIR}/image}"
AMBER_TOOLKIT="${AMBER_TOOLKIT:-${AMBER_DIR}/AMBER}"
MME_DIR="${MME_DIR:-${REPO_ROOT}/data/MME}"
MME_IMAGE_DIR="${MME_IMAGE_DIR:-${MME_DIR}/MME_Benchmark_release_version/MME_Benchmark}"
MME_QUESTIONS="${MME_QUESTIONS:-${MME_DIR}/test_merged_final.jsonl}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/transformers/src:${PYTHONPATH:-}"
export PYTHONSTARTUP="${REPO_ROOT}/causal_core/_python_startup.py"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

setup_env() {
  mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/slurm" "${SCORES_ROOT}"
  cd "${REPO_ROOT}"
}

setup_cluster() {
  setup_env
  if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load 2023 2>/dev/null || true
    module load Anaconda3/2023.07-2 2>/dev/null || true
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    _activate_conda_env() {
      local name="$1"
      if conda env list | awk '{print $1}' | grep -qx "${name}"; then
        conda activate "${name}"
        return 0
      fi
      return 1
    }
    if ! _activate_conda_env "${CONDA_ENV}"; then
      echo "Conda env '${CONDA_ENV}' not found." >&2
      echo "Create it with: bash scripts/setup_env.sh" >&2
      exit 1
    fi
  fi
  export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/transformers/src:${PYTHONPATH:-}"
  export PYTHONSTARTUP="${REPO_ROOT}/causal_core/_python_startup.py"
}

require_transformers_v5() {
  python - <<'PY'
import sys
import transformers
from packaging import version
if version.parse(transformers.__version__) < version.parse("5.0.0"):
    print(
        f"ERROR: transformers {transformers.__version__} is too old for Qwen3/InternVL "
        "(need >=5.0). Run: bash scripts/setup_env.sh",
        file=sys.stderr,
    )
    sys.exit(1)
PY
}

require_chair_deps() {
  if python - <<'PY'
import importlib
for pkg in ("nltk", "pycocotools", "pycocoevalcap", "google.protobuf", "tiktoken"):
    importlib.import_module(pkg)
PY
  then
    return 0
  fi
  echo "Installing eval dependencies (nltk, pycocotools, pycocoevalcap, protobuf, tiktoken)..." >&2
  pip install -q nltk pycocotools pycocoevalcap protobuf tiktoken
  python - <<'PY'
import importlib
for pkg in ("nltk", "pycocotools", "pycocoevalcap", "google.protobuf", "tiktoken"):
    importlib.import_module(pkg)
PY
}

# Qwen3 / InternVL CHAIR writers use: chall_alpha{alpha}_{method_name}.jsonl
chall_caption_path() {
  local out_dir="$1"
  local alpha="$2"
  local method="$3"
  echo "${out_dir}/chall_alpha${alpha}_${method}.jsonl"
}

run_chair_metrics() {
  local cap_file="$1"
  local save_path="$2"
  require_chair_deps
  python eval/chair.py \
    --cap_file "${cap_file}" \
    --coco_path "${COCO_DIR}/annotations" \
    --save_path "${save_path}" \
    --image_id_key image_id \
    --caption_key caption
}
