#!/bin/bash
#SBATCH --job-name=dev_bugatti_suv_30m
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_sam3D/logs/dev_bugatti_suv_30m_%j.out
#SBATCH --error=jobs_with_sam3D/logs/dev_bugatti_suv_30m_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

mkdir -p jobs_with_sam3D/logs
mkdir -p jobs_with_sam3D/outputs
mkdir -p jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

echo "======================================================"
echo "DEV SAM3D TEST: bugatti-centodieci.ply -> 'suv'"
echo "======================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOST=$(hostname)"
echo "START_TIME=$(date)"

python main.py \
  --config ./configs/base_config.yml \
  --mesh ./jobs_with_sam3D/meshes/bugatti-centodieci_upright_x90_wheels_down.ply \
  --text_prompt "suv" \
  --output_path ./jobs_with_sam3D/outputs/bugatti_to_suv_${SLURM_JOB_ID:-manual} \
  --epochs 300

echo "END_TIME=$(date)"
echo "Finished dev SAM3D run"
