#!/bin/bash
#SBATCH --job-name=dev_cross_animal
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_cross_animal_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_cross_animal_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 PAIR_ID"
  echo "PAIR_ID: 0 dachshund->golden, 1 golden->dachshund, 2 dachshund->cat, 3 cat->dachshund, 4 bulldog->cat, 5 cat->bulldog, 6 bulldog->dachshund, 7 dachshund->bulldog"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PAIR_ID="$1"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-2500}
RUN_TAG=${RUN_TAG:-topomatch_vcorr_dev_h100}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/topomatch_vcorr}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

bash "${BASE_DIR}/jobs/run_cross_animal_best_pair.sh" \
  "${PAIR_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
