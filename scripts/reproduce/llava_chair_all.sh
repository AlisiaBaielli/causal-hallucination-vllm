#!/usr/bin/env bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=llava_chair_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  source "${SLURM_SUBMIT_DIR}/scripts/_env.sh"
else
  source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
fi
setup_cluster

# Resume after partial runs: SKIP_EXISTING=1 sbatch scripts/reproduce/llava_chair_all.sh
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
bash "${REPO_ROOT}/scripts/reproduce/run_baselines_llava_chair.sh"
