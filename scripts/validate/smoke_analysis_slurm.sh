#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=smoke_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:15:00
#SBATCH --chdir=/home/abaielli/causal-hallucination-vlm
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -uo pipefail
source scripts/_env.sh
setup_cluster
bash scripts/validate/smoke_analysis.sh
