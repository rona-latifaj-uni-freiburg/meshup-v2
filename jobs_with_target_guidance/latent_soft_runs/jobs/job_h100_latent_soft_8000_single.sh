#!/bin/bash
#SBATCH --job-name=h100_pf_lat8k
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:30:00
#SBATCH --output=jobs_with_target_guidance/latent_soft_runs/logs/h100_pf_lat8k_%j.out
#SBATCH --error=jobs_with_target_guidance/latent_soft_runs/logs/h100_pf_lat8k_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 TASK_ID"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID="$1"
EPOCHS=${EPOCHS:-8000}
OUTPUT_ROOT=./jobs_with_target_guidance/latent_soft_runs/outputs_full
RUN_TAG=h100_full_single

bash ./jobs_with_target_guidance/latent_soft_runs/jobs/run_latent_soft_pair.sh \
  "${TASK_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
