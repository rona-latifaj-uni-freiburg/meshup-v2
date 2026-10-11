#!/bin/bash
#SBATCH --job-name=bundled_pf
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/bundled_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/bundled_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail
cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

# Computes a per-array-task row range from SLURM_ARRAY_TASK_ID and CHUNK_SIZE,
# then delegates to the existing generic array runner via its TASK_IDS_RANGE
# bundling path (job_dev_h100_sam3d_selected_partfield_4000_array.sh:18-40),
# so N sequential spec rows run inside one SLURM allocation instead of N
# separate array task entries. Exists purely to cut the number of discrete
# job-queue entries submitted to the shared cluster; does not change which
# rows run or how main.py is invoked per row.
CHUNK_SIZE=${CHUNK_SIZE:-2}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE - 1 ))
if [[ -n "${MAX_TASK_ID:-}" && ${END} -gt ${MAX_TASK_ID} ]]; then
  END=${MAX_TASK_ID}
fi
export TASK_IDS_RANGE="${START}-${END}"
echo "Bundled wrapper: array task ${SLURM_ARRAY_TASK_ID} -> spec rows ${TASK_IDS_RANGE}"

exec bash jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_sam3d_selected_partfield_4000_array.sh
