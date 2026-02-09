#!/bin/bash
#SBATCH --job-name=h100_motorcycle_centaur
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=logs/h100_motorcycle_centaur_%j.out
#SBATCH --error=logs/h100_motorcycle_centaur_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# CREATIVE: Motorcycle → Centaur
# Rider position + horse body from vehicle!

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_motorcycle_001/Aligned.obj \
  --output_path ./outputs/h100_motorcycle_centaur \
  --text_prompt "a centaur" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --use_cross_attn_loss \
  --cross_attn_weight 0.08 \
  --regularize_jacobians_weight 30000 \
  --epochs 15000
