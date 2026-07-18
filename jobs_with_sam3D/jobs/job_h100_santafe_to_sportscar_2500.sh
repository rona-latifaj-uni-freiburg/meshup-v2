#!/bin/bash
#SBATCH --job-name=h100_santafe_sportscar
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=jobs_with_sam3D/logs/h100_santafe_sportscar_%j.out
#SBATCH --error=jobs_with_sam3D/logs/h100_santafe_sportscar_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

echo "======================================================"
echo "H100 RUN: santa fe -> sports car"
echo "======================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOST=$(hostname)"
echo "START_TIME=$(date)"

python main.py \
  --config ./configs/base_config.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply \
  --text_prompt "sports car" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --dino_warmup_epochs 200 \
  --azim_min 300 \
  --azim_max 350 \
  --elev_max 35 \
  --output_path ./jobs_with_sam3D/outputs/santafe_to_sportscar_h100_${SLURM_JOB_ID:-manual} \
  --epochs 2500

echo "END_TIME=$(date)"
echo "Finished H100 santa fe -> sports car run"
