#!/bin/bash
#SBATCH --job-name=dc3d_obj_pair
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_object_pair_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_object_pair_%j.err

set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: sbatch $0 SOURCE_OBJECT TARGET_OBJECT MESH_VARIANT [EPOCHS]"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

SOURCE_OBJECT="$1"
TARGET_OBJECT="$2"
MESH_VARIANT="$3"
EPOCHS="${4:-${EPOCHS:-4000}}"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
RUN_TAG=${RUN_TAG:-densecorr3d_bucketbalanced_object_5k}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_bucketbalanced_object_5k_best_asym035}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

export BASE_DIR EPOCHS RUN_TAG OUTPUT_ROOT
if [[ -n "${PREPARED_DIR:-}" ]]; then
  export PREPARED_DIR
fi

bash "${BASE_DIR}/jobs/run_densecorr3d_object_pair.sh" \
  "${SOURCE_OBJECT}" \
  "${TARGET_OBJECT}" \
  "${MESH_VARIANT}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
