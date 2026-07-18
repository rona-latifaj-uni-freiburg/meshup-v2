#!/bin/bash
#SBATCH --job-name=densecorr3d_srcstable
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_srcstable_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_srcstable_%j.err

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
RUN_TAG=${RUN_TAG:-densecorr3d_srcstable_requested_5k}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_srcstable_requested_5k_best_asym035}
CHAIN_NEXT_SRCSTABLE=${CHAIN_NEXT_SRCSTABLE:-1}
NEXT_SUBMIT_RETRIES=${NEXT_SUBMIT_RETRIES:-30}
NEXT_SUBMIT_SLEEP=${NEXT_SUBMIT_SLEEP:-60}

case "${TASK_ID}" in
  0)
    PAIR_ID=8
    MESH_VARIANT=panther_cheetah_srcstable5002
    ;;
  1)
    PAIR_ID=5
    MESH_VARIANT=bear_panther_srcstable5002
    ;;
  2)
    PAIR_ID=9
    MESH_VARIANT=cheetah_panther_srcstable4992
    ;;
  3)
    PAIR_ID=10
    MESH_VARIANT=bear_elephant_srcstable5002
    ;;
  4)
    PAIR_ID=7
    MESH_VARIANT=elephant_moose_srcstable5000
    ;;
  5)
    PAIR_ID=1
    MESH_VARIANT=elephant_giraffe_srcstable5000
    ;;
  6)
    PAIR_ID=0
    MESH_VARIANT=giraffe_elephant_srcstable4996
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0..6."
    exit 2
    ;;
esac

submit_next_task() {
  local next_task="$1"
  local attempt=1
  local submit_output

  while true; do
    if submit_output=$(sbatch --parsable "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_srcstable_requested_single.sh" "${next_task}" 2>&1); then
      echo "${submit_output}"
      return 0
    fi

    echo "next_submit_attempt=${attempt} failed: ${submit_output}" >&2
    if (( attempt >= NEXT_SUBMIT_RETRIES )); then
      return 1
    fi
    sleep "${NEXT_SUBMIT_SLEEP}"
    attempt=$((attempt + 1))
  done
}

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

if [[ "${CHAIN_NEXT_SRCSTABLE}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  if (( TASK_ID < 6 )); then
    next_task=$((TASK_ID + 1))
    next_job=$(submit_next_task "${next_task}")
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_variant=${MESH_VARIANT}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "submitted_next_task=${next_task}"
      echo "next_job=${next_job}"
      echo "submitted_at=$(date)"
    } >> "${BASE_DIR}/reports/densecorr3d_srcstable_requested_chain.txt"
  else
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_variant=${MESH_VARIANT}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "chain_complete_at=$(date)"
    } >> "${BASE_DIR}/reports/densecorr3d_srcstable_requested_chain.txt"
  fi
fi
