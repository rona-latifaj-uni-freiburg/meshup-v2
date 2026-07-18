#!/bin/bash
#SBATCH --job-name=oracle_neckfix
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_oracle_neckfix_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_oracle_neckfix_%j.err

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 TASK_ID"
  echo "TASK_ID: 0 elephant->giraffe, 1 giraffe->elephant"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID="$1"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-10000}
RUN_TAG=${RUN_TAG:-oracle_neckfix_dev_h100}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/oracle_neckfix_20260702}
CHAIN_NEXT_ORACLE_NECKFIX=${CHAIN_NEXT_ORACLE_NECKFIX:-1}

case "${TASK_ID}" in
  0) PAIR_ID=0 ;;  # elephant -> giraffe
  1) PAIR_ID=1 ;;  # giraffe -> elephant
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0 or 1."
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs" "${BASE_DIR}/reports"

bash "${BASE_DIR}/jobs/run_oracle_neckfix_pair.sh" \
  "${PAIR_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

if [[ "${CHAIN_NEXT_ORACLE_NECKFIX}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  if (( TASK_ID < 1 )); then
    next_task=$((TASK_ID + 1))
    next_job=$(sbatch --parsable "${BASE_DIR}/jobs/job_dev_h100_oracle_neckfix_single.sh" "${next_task}")
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "submitted_next_task=${next_task}"
      echo "next_job=${next_job}"
      echo "submitted_at=$(date)"
    } >> "${BASE_DIR}/reports/oracle_neckfix_chain.txt"
  else
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_pair_id=${PAIR_ID}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "chain_complete_at=$(date)"
    } >> "${BASE_DIR}/reports/oracle_neckfix_chain.txt"
  fi
fi
