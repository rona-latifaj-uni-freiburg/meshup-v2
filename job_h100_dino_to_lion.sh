#!/bin/bash
#SBATCH --job-name=h100_dino_to_lion
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/h100_dino_to_lion_%j.out
#SBATCH --error=logs/h100_dino_to_lion_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# STRONG SEMANTIC CORRESPONDENCE: Dinosaur → Lion
# Both quadrupeds with similar body plan
# Head→head, 4 legs→4 legs, tail→tail, body→body

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-dinosaur_001/Aligned.obj \
  --output_path ./outputs/h100_dino_to_lion \
  --text_prompt "a majestic lion, standing, with mane" \
  --use_dino_loss \
  --dino_weight 0.15 \
  --regularize_jacobians_weight 50000 \
  --epochs 7000
