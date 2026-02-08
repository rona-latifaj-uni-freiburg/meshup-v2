#!/bin/bash
#SBATCH --job-name=h100_ball_to_hedgehog
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_ball_to_hedgehog_%j.out
#SBATCH --error=logs/h100_ball_to_hedgehog_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# NO PARTS → FULL ANIMAL: Ball → Hedgehog
# Simple sphere gains legs, face, spines - extreme emergence test

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-ball_001/Aligned.obj \
  --output_path ./outputs/h100_ball_to_hedgehog \
  --text_prompt "a hedgehog with spines" \
  --use_dino_loss \
  --dino_weight 0.05 \
  --regularize_jacobians_weight 20000 \
  --epochs 5000
