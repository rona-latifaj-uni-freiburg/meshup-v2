#!/bin/bash
#SBATCH --job-name=dm_animals
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/densematcher_runs/logs/dm_animals_%j.out
#SBATCH --error=jobs_with_target_guidance/densematcher_runs/logs/dm_animals_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

SOURCE_OBJECT=${1:-071b8_toy_animals_017}
TARGET_OBJECT=${2:-13cf7_toy_animals_055}
EPOCHS=${EPOCHS:-300}
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/densematcher_runs}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/animals_dev}
RUN_TAG=${RUN_TAG:-dev_h100}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

bash "${BASE_DIR}/jobs/run_densecorr3d_animal_pair.sh" \
  "${SOURCE_OBJECT}" \
  "${TARGET_OBJECT}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
