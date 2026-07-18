#!/bin/bash
#SBATCH --job-name=dev_topo_semreg_arr
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=4-7
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_topo_semreg_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_topo_semreg_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PAIR_ID=${SLURM_ARRAY_TASK_ID:-${1:-4}}
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}

bash "${BASE_DIR}/jobs/job_dev_h100_cross_animal_topomatch_semantic_moments_single.sh" "${PAIR_ID}"
