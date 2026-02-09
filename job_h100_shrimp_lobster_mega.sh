#!/bin/bash
#SBATCH --job-name=h100_shrimp_lobster_mega
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=logs/h100_shrimp_lobster_mega_%j.out
#SBATCH --error=logs/h100_shrimp_lobster_mega_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# MEGA DINO: Shrimp → Lobster (same family, perfect correspondence)
# Very high DINO + high reg - should preserve all parts perfectly

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-shrimp_010/Aligned.obj \
  --output_path ./outputs/h100_shrimp_lobster_mega \
  --text_prompt "a lobster" \
  --use_dino_loss \
  --dino_weight 0.3 \
  --regularize_jacobians_weight 70000 \
  --epochs 10000
