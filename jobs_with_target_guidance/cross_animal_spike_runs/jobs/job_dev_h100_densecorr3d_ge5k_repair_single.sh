#!/bin/bash
#SBATCH --job-name=densecorr3d_ge_fix
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_ge5k_repair_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_ge5k_repair_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PRESET=${1:-ge5k_profile_robust}
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-4000}
RUN_TAG=${RUN_TAG:-densecorr3d_ge5k_repair_${PRESET}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_giraffe_elephant_5k_repairs}
CHAIN_GE5K_REPAIR_PRESETS=${CHAIN_GE5K_REPAIR_PRESETS:-0}
CHAIN_REPORT=${CHAIN_REPORT:-${BASE_DIR}/reports/giraffe_to_elephant_5k_repair_chain.txt}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs" "${BASE_DIR}/reports"

export DENSECORR3D_REPAIR_PRESET="${PRESET}"
export CHAIN_GE5K_REPAIR_PRESETS

bash "${BASE_DIR}/jobs/run_densecorr3d_requested_animal_pair.sh" \
  0 \
  5k \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

if [[ "${CHAIN_GE5K_REPAIR_PRESETS}" == "1" && "${PRESET}" == "ge5k_profile_robust" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "CHAIN_GE5K_REPAIR_PRESETS=1 requested outside Slurm; not submitting follow-up." | tee -a "${CHAIN_REPORT}"
  else
    next_job_id=$(sbatch --parsable \
      --export=ALL,CHAIN_GE5K_REPAIR_PRESETS=0 \
      "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_ge5k_repair_single.sh" \
      ge5k_balanced_profile)
    printf '%s profile_robust_job=%s submitted_balanced_profile_job=%s\n' \
      "$(date -Is)" "${SLURM_JOB_ID}" "${next_job_id}" | tee -a "${CHAIN_REPORT}"
  fi
elif [[ "${CHAIN_GE5K_REPAIR_PRESETS}" == "1" && "${PRESET}" == "ge5k_refined32_local" ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "CHAIN_GE5K_REPAIR_PRESETS=1 requested outside Slurm; not submitting follow-up." | tee -a "${CHAIN_REPORT}"
  else
    next_job_id=$(sbatch --parsable \
      --export=ALL,CHAIN_GE5K_REPAIR_PRESETS=0 \
      "${BASE_DIR}/jobs/job_dev_h100_densecorr3d_ge5k_repair_single.sh" \
      ge5k_refined32_anchor)
    printf '%s refined32_local_job=%s submitted_refined32_anchor_job=%s\n' \
      "$(date -Is)" "${SLURM_JOB_ID}" "${next_job_id}" | tee -a "${CHAIN_REPORT}"
  fi
elif [[ "${CHAIN_GE5K_REPAIR_PRESETS}" == "1" ]]; then
  printf '%s completed_preset=%s job=%s\n' \
    "$(date -Is)" "${PRESET}" "${SLURM_JOB_ID:-manual}" | tee -a "${CHAIN_REPORT}"
fi
