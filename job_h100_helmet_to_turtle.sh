#!/bin/bash
#SBATCH --job-name=h100_helmet_to_turtle
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/h100_helmet_to_turtle_%j.out
#SBATCH --error=logs/h100_helmet_to_turtle_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# SEMANTIC CORRESPONDENCE: Helmet → Turtle
# Protective dome structure maps to shell
# Helmet dome → turtle shell, visor area → head emerges

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-helmet_002/Aligned.obj \
  --output_path ./outputs/h100_helmet_to_turtle \
  --text_prompt "a turtle" \
  --use_dino_loss \
  --dino_weight 0.12 \
  --regularize_jacobians_weight 35000 \
  --epochs 7000
