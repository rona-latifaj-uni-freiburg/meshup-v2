#!/bin/bash
#SBATCH --job-name=dc3d_pf0408
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/a100_densecorr3d_pf0408_chain_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/a100_densecorr3d_pf0408_chain_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID=${1:-0}
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
STOP_FILE=${STOP_FILE:-${BASE_DIR}/jobs/.stop_densecorr3d_pf0408_chain}
SPEC_FILE=${SPEC_FILE:-${BASE_DIR}/jobs/densecorr3d_all10_partfield_fair_pf0408_specs.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_partfield_guided_fair_all10_pf0408_4000_a100}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-densecorr3d_partfield_guided_fair_all10_a100}
AUTO_CHAIN=${AUTO_CHAIN:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
MAX_TASK=${MAX_TASK:-449}

mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/reports" "${OUTPUT_ROOT}"

if [[ -f "${STOP_FILE}" && "${TASK_ID}" != "0" ]]; then
  echo "DenseCorr3D PF4-PF8 chain stop file exists: ${STOP_FILE}"
  echo "Exiting chained successor task ${TASK_ID}; array runner is handling the remaining tasks."
  exit 0
fi

echo "DenseCorr3D full all-10 PF4-PF8 A100 chain"
echo "TASK_ID=${TASK_ID}/${MAX_TASK}"
echo "SPEC_FILE=${SPEC_FILE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "START_TIME=$(date)"

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
} >> "${BASE_DIR}/reports/densecorr3d_all10_pf0408_a100_chain.txt"

next_task=$((TASK_ID + 1))
if [[ "${AUTO_CHAIN}" == "1" && -n "${SLURM_JOB_ID:-}" && "${next_task}" -le "${MAX_TASK}" ]]; then
  next_job=$(sbatch --parsable \
    --dependency=afterany:${SLURM_JOB_ID} \
    --export=ALL,AUTO_CHAIN="${AUTO_CHAIN}",CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}",SPEC_FILE="${SPEC_FILE}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_TAG_PREFIX="${RUN_TAG_PREFIX}",MAX_TASK="${MAX_TASK}" \
    "${BASE_DIR}/jobs/job_gpu_a100_short_densecorr3d_partfield_fair_pf0408_chain.sh" "${next_task}")
  {
    echo "submitted_next_task=${next_task}"
    echo "submitted_next_job=${next_job}"
    echo "submitted_at=$(date)"
  } >> "${BASE_DIR}/reports/densecorr3d_all10_pf0408_a100_chain.txt"
fi

if [[ "${task_status}" != "0" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
  exit "${task_status}"
fi
