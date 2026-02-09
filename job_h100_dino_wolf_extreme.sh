#!/bin/bash
#SBATCH --job-name=h100_dino_wolf_extreme
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=logs/h100_dino_wolf_extreme_%j.out
#SBATCH --error=logs/h100_dino_wolf_extreme_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# EXTREME EPOCHS: Dinosaur → Wolf (both quadrupeds)
# 20k epochs, strong settings

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-dinosaur_016/Aligned.obj \
  --output_path ./outputs/h100_dino_wolf_extreme \
  --text_prompt "a wolf" \
  --use_dino_loss \
  --dino_weight 0.18 \
  --regularize_jacobians_weight 55000 \
  --epochs 20000
