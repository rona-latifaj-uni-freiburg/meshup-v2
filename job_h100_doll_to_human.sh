#!/bin/bash
#SBATCH --job-name=h100_doll_to_human
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_doll_to_human_%j.out
#SBATCH --error=logs/h100_doll_to_human_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# EXTENDED: Doll → Realistic Human (2 hours, 15000 epochs)
# Already humanoid! Best semantic correspondence baseline
# Perfect 1:1 mapping of all body parts

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-doll_001/Aligned.obj \
  --output_path ./outputs/h100_doll_to_human \
  --text_prompt "a realistic human person standing, anatomically correct proportions" \
  --use_dino_loss \
  --dino_weight 0.18 \
  --regularize_jacobians_weight 50000 \
  --epochs 15000
