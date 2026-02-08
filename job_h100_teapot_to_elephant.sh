#!/bin/bash
#SBATCH --job-name=h100_teapot_to_elephant
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_teapot_to_elephant_%j.out
#SBATCH --error=logs/h100_teapot_to_elephant_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# CREATIVE: Teapot → Elephant
# Spout becomes trunk! Handle becomes tail! Body stays round!

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-teapot_001/Aligned.obj \
  --output_path ./outputs/h100_teapot_to_elephant \
  --text_prompt "an elephant" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --regularize_jacobians_weight 35000 \
  --epochs 5000
