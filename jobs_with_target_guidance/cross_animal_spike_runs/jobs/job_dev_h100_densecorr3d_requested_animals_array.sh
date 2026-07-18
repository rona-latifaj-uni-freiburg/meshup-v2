#!/bin/bash
#SBATCH --job-name=densecorr3d_animals
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-15
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_requested_animals_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_requested_animals_%A_%a.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-4000}
RUN_TAG=${RUN_TAG:-densecorr3d_requested_animals}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_requested_animals_best_asym035}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

PAIR_ID=$((SLURM_ARRAY_TASK_ID / 2))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  MESH_VARIANT=full
else
  MESH_VARIANT=5k
fi

bash "${BASE_DIR}/jobs/run_densecorr3d_requested_animal_pair.sh" \
  "${PAIR_ID}" \
  "${MESH_VARIANT}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
