#!/bin/bash
#SBATCH --job-name=human_upright_v3
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/human_upright_v3_%j.out
#SBATCH --error=logs/human_upright_v3_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# Strategy: Lower regularization (allows more pose change) + weak DINO + explicit prompt
# Lower reg = more freedom to break quadruped pose

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/human_upright_v3 \
  --text_prompt "a human in T-pose standing" \
  --use_dino_loss \
  --dino_weight 0.05 \
  --regularize_jacobians_weight 10000 \
  --epochs 3300
