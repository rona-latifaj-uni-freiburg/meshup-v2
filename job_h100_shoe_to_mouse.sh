#!/bin/bash
#SBATCH --job-name=h100_shoe_to_mouse
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_shoe_to_mouse_%j.out
#SBATCH --error=logs/h100_shoe_to_mouse_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# NO TAIL → WITH TAIL: Shoe → Mouse
# Tests: Can a tailless object grow a tail? Toe becomes nose?

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-shoe_002/Aligned.obj \
  --output_path ./outputs/h100_shoe_to_mouse \
  --text_prompt "a cute gray mouse with long tail and big ears" \
  --use_dino_loss \
  --dino_weight 0.08 \
  --regularize_jacobians_weight 25000 \
  --epochs 5000
