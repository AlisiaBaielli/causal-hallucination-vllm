#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=chall_smoke_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/abaielli/causal-hallucination-vlm
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail
source scripts/_env.sh
setup_cluster

bash scripts/validate/smoke_all.sh
