#!/bin/bash
# Generic SLURM wrapper for the reproduce driver. Submitted by submit_all.sh with:
#   sbatch --export=ALL,RMODEL=llava,RBENCH=chair --job-name=repro_llava_chair _job.sh
# RKIND=quality runs the caption-quality metrics instead of a benchmark.
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
# Working dir defaults to the SLURM submission directory; submit from the repo root.
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -uo pipefail
source scripts/_env.sh
setup_cluster

if [[ "${RKIND:-bench}" == "quality" ]]; then
  bash scripts/reproduce/quality.sh "${RMODEL}"
else
  bash scripts/reproduce/run.sh "${RMODEL}" "${RBENCH}"
fi
