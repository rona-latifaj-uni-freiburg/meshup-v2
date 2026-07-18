#!/bin/bash
#SBATCH --job-name=densecorr3d_bb
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_bucketbalanced_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_bucketbalanced_%j.err

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 TASK_ID"
  echo "TASK_ID: 0 panther->cheetah, 1 bear->panther, 2 cheetah->panther, 3 bear->elephant, 4 elephant->moose, 5 elephant->giraffe, 6 giraffe->elephant"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID="$1"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-4000}
RUN_TAG=${RUN_TAG:-densecorr3d_bucketbalanced_requested_5k}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_bucketbalanced_requested_5k_best_asym035}
CHAIN_NEXT_BUCKETBALANCED=${CHAIN_NEXT_BUCKETBALANCED:-1}

case "${TASK_ID}" in
  0)
    PAIR_ID=8
    MESH_VARIANT=panther_cheetah_bucketavg4999
    ;;
  1)
    PAIR_ID=5
    MESH_VARIANT=bear_panther_bucketavg5002
    ;;
  2)
    PAIR_ID=9
    MESH_VARIANT=panther_cheetah_bucketavg4999
    ;;
  3)
    PAIR_ID=10
    MESH_VARIANT=bear_elephant_bucketavg5001
    ;;
  4)
    PAIR_ID=7
    MESH_VARIANT=elephant_moose_bucketavg5001
    ;;
  5)
    PAIR_ID=1
    MESH_VARIANT=elephant_giraffe_bucketavg4998
    ;;
  6)
    PAIR_ID=0
    MESH_VARIANT=elephant_giraffe_bucketavg4998
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0..6."
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs" "${BASE_DIR}/reports"

export BASE_DIR EPOCHS RUN_TAG OUTPUT_ROOT
if [[ -n "${PREPARED_DIR:-}" ]]; then
  export PREPARED_DIR
fi

bash "${BASE_DIR}/jobs/run_densecorr3d_requested_animal_pair.sh" \
  "${PAIR_ID}" \
  "${MESH_VARIANT}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

if [[ "${CHAIN_NEXT_BUCKETBALANCED}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  if (( TASK_ID < 6 )); then
    next_task=$((TASK_ID + 1))
    next_job=$(sbatch --parsable "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_bucketbalanced_requested_single.sh" "${next_task}")
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_variant=${MESH_VARIANT}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "submitted_next_task=${next_task}"
      echo "next_job=${next_job}"
      echo "submitted_at=$(date)"
    } >> "${BASE_DIR}/reports/densecorr3d_bucketbalanced_requested_chain.txt"
  else
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_variant=${MESH_VARIANT}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "chain_complete_at=$(date)"
    } >> "${BASE_DIR}/reports/densecorr3d_bucketbalanced_requested_chain.txt"
  fi
fi
