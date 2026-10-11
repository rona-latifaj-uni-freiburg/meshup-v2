#!/bin/bash
#SBATCH --job-name=tosca_vcorr_chain
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/tosca_vcorr_chain_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/tosca_vcorr_chain_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

TASK_ID=${1:-0}
MAX_TASK=${MAX_TASK:-22}

echo "TOSCA correspondence chain task ${TASK_ID}/${MAX_TASK}"
echo "START_TIME=$(date)"

bash jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_tosca_correspondence_guided_array.sh "${TASK_ID}"

echo "END_TIME=$(date)"

if [[ "${AUTO_CHAIN:-1}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  next_task=$((TASK_ID + 1))
  if (( next_task <= MAX_TASK )); then
    next_job=$(sbatch --parsable \
      --export=ALL,AUTO_CHAIN=1,MAX_TASK="${MAX_TASK}" \
      jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_tosca_correspondence_guided_chain.sh \
      "${next_task}")
    mkdir -p jobs_with_target_guidance/cross_animal_spike_runs/reports
    {
      echo "finished_task=${TASK_ID}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "submitted_next_task=${next_task}"
      echo "next_job=${next_job}"
      echo "submitted_at=$(date)"
    } > "jobs_with_target_guidance/cross_animal_spike_runs/reports/tosca_vcorr_chain_after_task_${TASK_ID}.txt"
  else
    echo "TOSCA correspondence chain complete."
  fi
fi
