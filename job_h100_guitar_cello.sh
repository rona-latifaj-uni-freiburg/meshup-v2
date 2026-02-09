#!/bin/bash
#SBATCH --job-name=h100_guitar_cello
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_guitar_cello_%j.out
#SBATCH --error=logs/h100_guitar_cello_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# STRONG CORRESPONDENCE: Guitar → Cello
# String instruments, nearly perfect part mapping

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-guitar_001/Aligned.obj \
  --output_path ./outputs/h100_guitar_cello \
  --text_prompt "a cello" \
  --use_dino_loss \
  --dino_weight 0.22 \
  --regularize_jacobians_weight 55000 \
  --epochs 12000
