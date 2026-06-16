#!/bin/bash
#SBATCH --job-name=dev_art_anim
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_artur_animal_8k_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_artur_animal_8k_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch $0 PAIR_ID VARIANT_ID"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PAIR_ID="$1"
VARIANT_ID="$2"
EPOCHS=${EPOCHS:-8000}
OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_animals_8k
RUN_TAG=dev_h100_animal_single
export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-1000.0}

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_animal_pair_8k.sh \
  "${PAIR_ID}" \
  "${VARIANT_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
