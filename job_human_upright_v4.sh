#!/bin/bash
#SBATCH --job-name=human_upright_v4
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/human_upright_v4_%j.out
#SBATCH --error=logs/human_upright_v4_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# Strategy: No DINO (baseline with good prompt) + lower reg
# See if just the prompt can guide to upright pose without DINO preserving quadruped

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/human_upright_v4 \
  --text_prompt "a person standing tall on feet" \
  --regularize_jacobians_weight 15000 \
  --epochs 3300
