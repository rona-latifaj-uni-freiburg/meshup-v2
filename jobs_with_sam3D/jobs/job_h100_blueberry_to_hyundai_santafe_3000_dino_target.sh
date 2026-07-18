#!/bin/bash
#SBATCH --job-name=h100_blueberry_hyundai_santafe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=jobs_with_sam3D/logs/h100_blueberry_hyundai_santafe_%j.out
#SBATCH --error=jobs_with_sam3D/logs/h100_blueberry_hyundai_santafe_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

echo "======================================================"
echo "H100 RUN: blueberry(opel corsa size) -> hyundai santa fe (bigger SUV)"
echo "======================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "HOST=$(hostname)"
echo "START_TIME=$(date)"

python main.py \
  --config ./configs/base_config.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply \
  --text_prompt "a Hyundai Santa Fe, a larger SUV" \
  --use_dino_loss \
  --dino_weight 0.12 \
  --dino_warmup_epochs 250 \
  --use_target_mesh_guidance \
  --target_mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply \
  --target_mesh_weight 0.7 \
  --target_mesh_warmup_epochs 250 \
  --target_mesh_n_azimuths 12 \
  --target_mesh_n_elevations 3 \
  --azim_min 300 \
  --azim_max 350 \
  --elev_max 35 \
  --output_path ./jobs_with_sam3D/outputs/blueberry_to_hyundai_santafe_h100_${SLURM_JOB_ID:-manual} \
  --epochs 3500

echo "END_TIME=$(date)"
echo "Finished H100 blueberry -> Hyundai Santa Fe run"
