#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

python experiments/baselines/vcd_m3id_internvl.py --help >/dev/null 2>&1 || true
echo "Use experiments/chair/internvl.py --use_vcd / --use_m3id for InternVL baselines."
