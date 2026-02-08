#!/bin/bash
#SBATCH --job-name=h100_car_to_beetle
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_car_to_beetle_%j.out
#SBATCH --error=logs/h100_car_to_beetle_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# CREATIVE: Toy car → Beetle insect
# Wheels become legs, body stays compact, gains antennae

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_car_001/Aligned.obj \
  --output_path ./outputs/h100_car_to_beetle \
  --text_prompt "a shiny beetle insect with six legs and antennae" \
  --use_dino_loss \
  --dino_weight 0.08 \
  --regularize_jacobians_weight 30000 \
  --epochs 5000
