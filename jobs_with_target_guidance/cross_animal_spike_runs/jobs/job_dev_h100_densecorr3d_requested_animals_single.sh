#!/bin/bash
#SBATCH --job-name=densecorr3d_animal
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_requested_animal_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_requested_animal_%j.err

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch $0 PAIR_ID MESH_VARIANT"
  echo "PAIR_ID: 0 giraffe->elephant, 1 elephant->giraffe, 2 bear->cheetah, 3 cheetah->bear, 4 panther->bear, 5 bear->panther, 6 moose->elephant, 7 elephant->moose"
  echo "MESH_VARIANT: full, 5k, or any prepared suffix such as bucketaverage4998"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PAIR_ID="$1"
MESH_VARIANT="$2"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-4000}
RUN_TAG=${RUN_TAG:-densecorr3d_requested_animals}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_requested_animals_best_asym035}
CHAIN_NEXT_DENSECORR3D_ANIMALS=${CHAIN_NEXT_DENSECORR3D_ANIMALS:-0}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

export BASE_DIR EPOCHS RUN_TAG OUTPUT_ROOT CHAIN_NEXT_DENSECORR3D_ANIMALS
if [[ -n "${PREPARED_DIR:-}" ]]; then
  export PREPARED_DIR
fi

bash "${BASE_DIR}/jobs/run_densecorr3d_requested_animal_pair.sh" \
  "${PAIR_ID}" \
  "${MESH_VARIANT}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

if [[ "${MESH_VARIANT}" == "full" || "${MESH_VARIANT}" == "5k" ]]; then
  task_id=$((PAIR_ID * 2))
  if [[ "${MESH_VARIANT}" == "5k" ]]; then
    task_id=$((task_id + 1))
  fi
else
  task_id=-1
fi

if [[ "${CHAIN_NEXT_DENSECORR3D_ANIMALS}" == "1" && -n "${SLURM_JOB_ID:-}" && "${task_id}" -ge 0 ]]; then
  mkdir -p "${BASE_DIR}/reports"
  if (( task_id < 15 )); then
    next_task=$((task_id + 1))
    next_pair=$((next_task / 2))
    if (( next_task % 2 == 0 )); then
      next_variant=full
    else
      next_variant=5k
    fi
    next_job=$(sbatch --parsable "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_requested_animals_single.sh" "${next_pair}" "${next_variant}")
    {
      echo "finished_task=${task_id}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_variant=${MESH_VARIANT}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "submitted_next_task=${next_task}"
      echo "submitted_next_pair_id=${next_pair}"
      echo "submitted_next_variant=${next_variant}"
      echo "next_job=${next_job}"
      echo "submitted_at=$(date)"
    } >> "${BASE_DIR}/reports/densecorr3d_requested_animals_chain.txt"
  else
    {
      echo "finished_task=${task_id}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_variant=${MESH_VARIANT}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "chain_complete_at=$(date)"
    } >> "${BASE_DIR}/reports/densecorr3d_requested_animals_chain.txt"
  fi
fi
