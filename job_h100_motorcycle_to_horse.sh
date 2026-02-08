#!/bin/bash
#SBATCH --job-name=h100_motorcycle_to_horse
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/h100_motorcycle_to_horse_%j.out
#SBATCH --error=logs/h100_motorcycle_to_horse_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# CONCEPTUAL SEMANTIC CORRESPONDENCE: Motorcycle → Horse
# Both are "mounts" - rideable with similar riding position
# Wheels→legs, seat→back, handlebars→head/neck, frame→body

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_motorcycle_001/Aligned.obj \
  --output_path ./outputs/h100_motorcycle_to_horse \
  --text_prompt "a horse standing, side view, realistic" \
  --use_dino_loss \
  --dino_weight 0.12 \
  --regularize_jacobians_weight 40000 \
  --epochs 7000
