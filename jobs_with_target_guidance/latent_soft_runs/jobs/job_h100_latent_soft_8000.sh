#!/bin/bash
#SBATCH --job-name=h100_pf_lat8k
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:30:00
#SBATCH --array=0-5
#SBATCH --output=jobs_with_target_guidance/latent_soft_runs/logs/h100_pf_lat8k_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/latent_soft_runs/logs/h100_pf_lat8k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
EPOCHS=${EPOCHS:-8000}
OUTPUT_ROOT=./jobs_with_target_guidance/latent_soft_runs/outputs_full
RUN_TAG=h100_full

bash ./jobs_with_target_guidance/latent_soft_runs/jobs/run_latent_soft_pair.sh \
  "${TASK_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
