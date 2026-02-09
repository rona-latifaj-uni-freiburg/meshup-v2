#!/bin/bash
#SBATCH --job-name=h100_plane_bird_strong_dino
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_plane_bird_strong_dino_%j.out
#SBATCH --error=logs/h100_plane_bird_strong_dino_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# STRONG DINO: Plane → Bird
# High DINO weight to really enforce wing→wing correspondence

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_plane_007/Aligned.obj \
  --output_path ./outputs/h100_plane_bird_strong_dino \
  --text_prompt "an eagle with wings spread" \
  --use_dino_loss \
  --dino_weight 0.25 \
  --regularize_jacobians_weight 60000 \
  --epochs 12000
