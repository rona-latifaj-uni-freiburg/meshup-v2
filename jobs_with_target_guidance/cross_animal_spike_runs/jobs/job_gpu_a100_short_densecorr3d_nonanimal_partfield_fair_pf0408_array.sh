#!/bin/bash
#SBATCH --job-name=dc3d_nonanim_pf0408
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-999%20
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/a100_densecorr3d_nonanimal_pf0408_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/a100_densecorr3d_nonanimal_pf0408_%A_%a.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
SPEC_FILE=${SPEC_FILE:-${BASE_DIR}/jobs/densecorr3d_nonanimals_test_partfield_fair_pf0408_specs.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-densecorr3d_nonanimals_test_partfield_guided_fair_a100}
TASK_OFFSET=${TASK_OFFSET:-0}
RAW_TASK_ID=${1:-${SLURM_ARRAY_TASK_ID:-}}
COMPACT_OUTPUT_NAMES=${COMPACT_OUTPUT_NAMES:-1}

mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/reports" "${OUTPUT_ROOT}"

if [[ -z "${RAW_TASK_ID}" ]]; then
  echo "Missing task id. Submit as a Slurm array or pass a task id."
  exit 2
fi

TASK_ID=$((RAW_TASK_ID + TASK_OFFSET))

echo "DenseCorr3D non-animal test PartField PF4-PF8 A100 array"
echo "TASK_ID=${TASK_ID}/2959 (offset ${TASK_OFFSET} + array task ${RAW_TASK_ID})"
echo "SPEC_FILE=${SPEC_FILE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "START_TIME=$(date)"

SPEC_FILE="${SPEC_FILE}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_TAG_PREFIX="${RUN_TAG_PREFIX}" \
COMPACT_OUTPUT_NAMES="${COMPACT_OUTPUT_NAMES}" \
bash "${BASE_DIR}/jobs/job_dev_h100_sam3d_selected_partfield_4000_array.sh" "${TASK_ID}"

{
  echo "task_id=${TASK_ID}"
  echo "job_id=${SLURM_JOB_ID:-local}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "finished_at=$(date)"
} >> "${BASE_DIR}/reports/densecorr3d_nonanimals_test_pf0408_a100_array.txt"
