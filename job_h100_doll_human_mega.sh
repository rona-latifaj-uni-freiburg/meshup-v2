#!/bin/bash
#SBATCH --job-name=h100_doll_human_mega
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/h100_doll_human_mega_%j.out
#SBATCH --error=logs/h100_doll_human_mega_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# MEGA: Doll → Human (3 hours, 25k epochs)
# Best correspondence test - already humanoid!

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-doll_002/Aligned.obj \
  --output_path ./outputs/h100_doll_human_mega \
  --text_prompt "a human" \
  --use_dino_loss \
  --dino_weight 0.25 \
  --use_cross_attn_loss \
  --cross_attn_weight 0.1 \
  --regularize_jacobians_weight 60000 \
  --epochs 25000
