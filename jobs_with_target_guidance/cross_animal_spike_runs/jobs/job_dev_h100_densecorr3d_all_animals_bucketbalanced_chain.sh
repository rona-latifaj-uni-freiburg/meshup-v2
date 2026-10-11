#!/bin/bash
#SBATCH --job-name=dc3d_all_chain
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_all_animals_chain_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_all_animals_chain_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID=${1:-0}
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
ANIMAL_SPLIT=${ANIMAL_SPLIT:-all}
PREPARED_DIR=${PREPARED_DIR:-./jobs_with_target_guidance/densematcher_runs/prepared/all_animals_20260803}
EPOCHS=${EPOCHS:-4000}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_all_animals_bucketbalanced_5k_best_asym035}
RUN_TAG=${RUN_TAG:-densecorr3d_all_animals_bucketbalanced_5k}
AUTO_CHAIN=${AUTO_CHAIN:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}

case "${ANIMAL_SPLIT}" in
  all) MAX_TASK=${MAX_TASK:-89} ;;
  test|train) MAX_TASK=${MAX_TASK:-19} ;;
  *)
    echo "Unknown ANIMAL_SPLIT=${ANIMAL_SPLIT}. Expected all, test, or train."
    exit 2
    ;;
esac

mkdir -p "${BASE_DIR}/reports" "${BASE_DIR}/logs" "${OUTPUT_ROOT}"

set +e
BASE_DIR="${BASE_DIR}" \
ANIMAL_SPLIT="${ANIMAL_SPLIT}" \
PREPARED_DIR="${PREPARED_DIR}" \
EPOCHS="${EPOCHS}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_TAG="${RUN_TAG}" \
bash "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_all_animals_bucketbalanced_array.sh" "${TASK_ID}"
task_status=$?
set -e

{
  echo "task_id=${TASK_ID}"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "status=${task_status}"
  echo "animal_split=${ANIMAL_SPLIT}"
  echo "finished_at=$(date)"
} >> "${BASE_DIR}/reports/densecorr3d_all_animals_bucketbalanced_chain_${ANIMAL_SPLIT}.txt"

next_task=$((TASK_ID + 1))
if [[ "${AUTO_CHAIN}" == "1" && -n "${SLURM_JOB_ID:-}" && "${next_task}" -le "${MAX_TASK}" ]]; then
  next_job=$(sbatch --parsable \
    --dependency=afterany:${SLURM_JOB_ID} \
    --export=ALL,AUTO_CHAIN="${AUTO_CHAIN}",CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}",ANIMAL_SPLIT="${ANIMAL_SPLIT}",PREPARED_DIR="${PREPARED_DIR}",EPOCHS="${EPOCHS}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_TAG="${RUN_TAG}",MAX_TASK="${MAX_TASK}" \
    "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_all_animals_bucketbalanced_chain.sh" "${next_task}")
  {
    echo "submitted_next_task=${next_task}"
    echo "submitted_next_job=${next_job}"
    echo "submitted_at=$(date)"
  } >> "${BASE_DIR}/reports/densecorr3d_all_animals_bucketbalanced_chain_${ANIMAL_SPLIT}.txt"
fi

if [[ "${task_status}" != "0" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
  exit "${task_status}"
fi
