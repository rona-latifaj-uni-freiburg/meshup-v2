#!/bin/bash
#SBATCH --job-name=h100_teddy_gorilla
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=logs/h100_teddy_gorilla_%j.out
#SBATCH --error=logs/h100_teddy_gorilla_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# STRONG CORRESPONDENCE: Teddy Bear → Gorilla
# Both have similar sitting pose, arms, legs, round body

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-teddy_bear_003/Aligned.obj \
  --output_path ./outputs/h100_teddy_gorilla \
  --text_prompt "a gorilla" \
  --use_dino_loss \
  --dino_weight 0.2 \
  --regularize_jacobians_weight 50000 \
  --epochs 15000
