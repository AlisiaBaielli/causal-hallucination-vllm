#!/usr/bin/env bash
# Caption-quality / alignment metrics (METEOR + CLIPScore) over the reproduce
# CHAIR captions for one model. Requires scripts/reproduce/run.sh <model> chair
# to have produced captions first.
#
#   bash scripts/reproduce/quality.sh <model>     model: llava | qwen3 | internvl
set -uo pipefail

MODEL="${1:?usage: quality.sh <model>}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster
require_chair_deps

CHAIR_BASE="${OUT_ROOT}/reproduce/${MODEL}_chair"
OUT="${OUT_ROOT}/reproduce/${MODEL}_quality"
mkdir -p "${OUT}"
COCO_CAPS="${COCO_DIR}/annotations/captions_val2014.json"
COCO_IMG="${COCO_DIR}/val2014"
COCO_ANNO="${COCO_DIR}/annotations/instances_val2014.json"

pairs=()
for m in vanilla vcd m3id only only_eic chall; do
  cap="$(ls -t "${CHAIR_BASE}/${m}"/*.jsonl 2>/dev/null | head -1)"
  [[ -n "${cap}" ]] && pairs+=("${m}:${cap}")
done
if [[ ${#pairs[@]} -eq 0 ]]; then
  echo "No CHAIR captions under ${CHAIR_BASE}; run: bash scripts/reproduce/run.sh ${MODEL} chair" >&2
  exit 1
fi

echo "=== ${MODEL} METEOR ==="
python experiments/analysis/compute_meteor.py --caption_files "${pairs[@]}" \
  --coco_captions "${COCO_CAPS}" --out_path "${OUT}/meteor.json"

echo "=== ${MODEL} CLIPScore ==="
python experiments/analysis/compute_clipscore.py --caption_files "${pairs[@]}" \
  --image_dir "${COCO_IMG}" --anno_path "${COCO_ANNO}" --format jsonl \
  --out_path "${OUT}/clipscore.json"

echo "Quality metrics under ${OUT}"
