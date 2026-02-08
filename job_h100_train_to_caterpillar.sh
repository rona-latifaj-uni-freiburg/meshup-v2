#!/bin/bash
#SBATCH --job-name=h100_train_to_caterpillar
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/h100_train_to_caterpillar_%j.out
#SBATCH --error=logs/h100_train_to_caterpillar_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# EXCELLENT SEMANTIC CORRESPONDENCE: Train → Caterpillar
# Segmented body structure maps perfectly!
# Train cars → body segments, wheels → legs, front → head

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_train_001/Aligned.obj \
  --output_path ./outputs/h100_train_to_caterpillar \
  --text_prompt "a green caterpillar with many legs and segments" \
  --use_dino_loss \
  --dino_weight 0.15 \
  --regularize_jacobians_weight 45000 \
  --epochs 7000
