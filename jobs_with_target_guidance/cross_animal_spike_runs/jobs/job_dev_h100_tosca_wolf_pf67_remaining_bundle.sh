#!/bin/bash
#SBATCH --job-name=tosca_wolf_pf67b
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/tosca_wolf_pf67_remaining_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/tosca_wolf_pf67_remaining_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

export SPEC_FILE=${SPEC_FILE:-jobs_with_target_guidance/cross_animal_spike_runs/jobs/tosca_wolf_partfield_pilot_specs.tsv}
export OUTPUT_ROOT=${OUTPUT_ROOT:-./jobs_with_target_guidance/cross_animal_spike_runs/outputs/tosca_partfield_wolf_pilot}
export RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-tosca_wolf}
export RUN_EVAL=${RUN_EVAL:-0}

# Remaining PF6/PF7 rows after row 0 completed in job 6165545.
if [[ -z "${TASK_IDS:-}" ]]; then
  export TASK_IDS="1 3 4 6 7"
fi

echo "Running remaining bundled TOSCA wolf PF6/PF7 tasks: ${TASK_IDS}"
echo "SPEC_FILE=${SPEC_FILE}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "START_TIME=$(date)"

bash jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_sam3d_selected_partfield_4000_array.sh

echo "END_TIME=$(date)"
