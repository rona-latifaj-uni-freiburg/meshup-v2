#!/bin/bash
#SBATCH --job-name=h100_shrimp_to_lobster
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/h100_shrimp_to_lobster_%j.out
#SBATCH --error=logs/h100_shrimp_to_lobster_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# PERFECT SEMANTIC CORRESPONDENCE: Shrimp → Lobster
# Same family! Nearly identical body plan
# Antennae→antennae, claws→claws, tail→tail, legs→legs

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-shrimp_001/Aligned.obj \
  --output_path ./outputs/h100_shrimp_to_lobster \
  --text_prompt "a red lobster with large claws and antennae" \
  --use_dino_loss \
  --dino_weight 0.2 \
  --regularize_jacobians_weight 55000 \
  --epochs 6500
