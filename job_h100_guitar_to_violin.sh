#!/bin/bash
#SBATCH --job-name=h100_guitar_to_violin
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:20:00
#SBATCH --output=logs/h100_guitar_to_violin_%j.out
#SBATCH --error=logs/h100_guitar_to_violin_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# STRONG SEMANTIC CORRESPONDENCE: Guitar → Violin
# String instruments with clear part mapping
# Body→body, neck→neck, headstock→scroll, strings→strings

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-guitar_001/Aligned.obj \
  --output_path ./outputs/h100_guitar_to_violin \
  --text_prompt "a classical violin" \
  --use_dino_loss \
  --dino_weight 0.18 \
  --regularize_jacobians_weight 50000 \
  --epochs 6000
