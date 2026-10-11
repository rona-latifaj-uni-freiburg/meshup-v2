#!/bin/bash
#SBATCH --job-name=dc3d_pf_fair
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_pf_fair_chain_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_pf_fair_chain_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID=${1:-0}
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
SPEC_FILE=${SPEC_FILE:-${BASE_DIR}/jobs/densecorr3d_selected_partfield_4000_specs.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_partfield_guided_fair_4000}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-densecorr3d_partfield_guided_fair}
AUTO_CHAIN=${AUTO_CHAIN:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
MAX_TASK=${MAX_TASK:-11}

mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/reports" "${OUTPUT_ROOT}"

set +e
SPEC_FILE="${SPEC_FILE}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_TAG_PREFIX="${RUN_TAG_PREFIX}" \
bash "${BASE_DIR}/jobs/job_dev_h100_sam3d_selected_partfield_4000_array.sh" "${TASK_ID}"
task_status=$?
set -e

{
  echo "task_id=${TASK_ID}"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "status=${task_status}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "finished_at=$(date)"
} >> "${BASE_DIR}/reports/densecorr3d_partfield_guided_fair_chain.txt"

next_task=$((TASK_ID + 1))
if [[ "${AUTO_CHAIN}" == "1" && -n "${SLURM_JOB_ID:-}" && "${next_task}" -le "${MAX_TASK}" ]]; then
  next_job=$(sbatch --parsable \
    --dependency=afterany:${SLURM_JOB_ID} \
    --export=ALL,AUTO_CHAIN="${AUTO_CHAIN}",CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}",SPEC_FILE="${SPEC_FILE}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_TAG_PREFIX="${RUN_TAG_PREFIX}",MAX_TASK="${MAX_TASK}" \
    "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_partfield_guided_fair_chain.sh" "${next_task}")
  {
    echo "submitted_next_task=${next_task}"
    echo "submitted_next_job=${next_job}"
    echo "submitted_at=$(date)"
  } >> "${BASE_DIR}/reports/densecorr3d_partfield_guided_fair_chain.txt"
fi

if [[ "${task_status}" != "0" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
  exit "${task_status}"
fi
