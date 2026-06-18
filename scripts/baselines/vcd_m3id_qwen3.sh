#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

python experiments/baselines/vcd_m3id_qwen3.py --help >/dev/null 2>&1 || true
echo "Use experiments/chair/qwen3.py --use_vcd / --use_m3id for Qwen3 baselines."
