#!/bin/bash
#SBATCH --job-name=dev_outlier_4807067
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_outlier_4807067_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_outlier_4807067_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python \
  jobs_with_target_guidance/analyze_displacement_outliers.py \
  jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_global_chamfer_only_2500ep_4807067
