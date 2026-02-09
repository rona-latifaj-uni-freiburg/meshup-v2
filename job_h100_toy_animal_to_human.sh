#!/bin/bash
#SBATCH --job-name=h100_gorilla_to_human
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_gorilla_to_human_%j.out
#SBATCH --error=logs/h100_gorilla_to_human_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# EXTENDED: Toy Animal (Ape-like) → Human (2 hours, 15000 epochs)
# Much better starting point - already semi-bipedal, has hands!
# Arms→arms, legs→legs, face→face, hands→hands

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_animals_001/Aligned.obj \
  --output_path ./outputs/h100_toy_animal_to_human \
  --text_prompt "a human person standing upright, T-pose, realistic proportions" \
  --use_dino_loss \
  --dino_weight 0.12 \
  --regularize_jacobians_weight 30000 \
  --epochs 15000
