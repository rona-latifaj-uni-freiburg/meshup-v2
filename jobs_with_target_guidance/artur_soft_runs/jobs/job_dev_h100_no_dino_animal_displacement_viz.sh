#!/bin/bash
#SBATCH --job-name=dev_nd_disp
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_nd_disp_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_nd_disp_%j.err

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 RUN_DIR [RUN_DIR ...]"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/meshup_mplconfig_${SLURM_JOB_ID:-manual}
mkdir -p "${MPLCONFIGDIR}" jobs_with_target_guidance/artur_soft_runs/logs

EPOCHS=(
  1
  10 20 30 40 50 60 70 80 90 100
  250 500 750 1000 1250 1500 1750 2000
  2250 2500 2750 3000 3250 3500 3750 4000
)

/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python \
  jobs_with_target_guidance/visualize_run_displacements.py \
  "$@" \
  --epochs "${EPOCHS[@]}"
