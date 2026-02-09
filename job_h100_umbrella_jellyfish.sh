#!/bin/bash
#SBATCH --job-name=h100_umbrella_jellyfish
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_umbrella_jellyfish_%j.out
#SBATCH --error=logs/h100_umbrella_jellyfish_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# PERFECT SHAPE MATCH: Umbrella → Jellyfish
# Dome→bell, handle→tentacles (if present)

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-umbrella_001/Aligned.obj \
  --output_path ./outputs/h100_umbrella_jellyfish \
  --text_prompt "a jellyfish" \
  --use_dino_loss \
  --dino_weight 0.12 \
  --use_cross_attn_loss \
  --cross_attn_weight 0.1 \
  --regularize_jacobians_weight 35000 \
  --epochs 12000
