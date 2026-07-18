#!/bin/bash
#SBATCH --job-name=dev_bugatti6k_rally_suv
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_sam3D/logs/dev_bugatti6k_rally_suv_%j.out
#SBATCH --error=jobs_with_sam3D/logs/dev_bugatti6k_rally_suv_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

mkdir -p jobs_with_sam3D/logs
mkdir -p jobs_with_sam3D/outputs
mkdir -p jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

echo "======================================================"
echo "DEV SAM3D TEST: decimated bugatti -> rally SUV"
echo "======================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOST=$(hostname)"
echo "START_TIME=$(date)"

python main.py \
  --config ./configs/base_config.yml \
  --mesh ./jobs_with_sam3D/meshes/decimated/bugatti-centodieci_upright_wheels_down_6kverts.ply \
  --text_prompt "Ferrari F40" \
  --azim_min 300 \
  --azim_max 350 \
  --elev_max 35 \
  --output_path ./jobs_with_sam3D/outputs/bugatti6k_to_rally_suv_${SLURM_JOB_ID:-manual} \
  --epochs 300

echo "END_TIME=$(date)"
echo "Finished dev decimated Bugatti run"
