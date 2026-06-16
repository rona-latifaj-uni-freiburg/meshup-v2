#!/bin/bash
#SBATCH --job-name=dev_disp_early
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_disp_early_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_disp_early_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/meshup_mplconfig_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}" jobs_with_target_guidance/artur_soft_runs/logs

RUN_DIRS=(jobs_with_target_guidance/artur_soft_runs/outputs_dev/*)

/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python \
  jobs_with_target_guidance/visualize_run_displacements.py \
  "${RUN_DIRS[@]}" \
  --epochs 2 3 4 5 6 7 8 9 10 20 30 40 50
