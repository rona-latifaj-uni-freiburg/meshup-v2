#!/bin/bash
#SBATCH --job-name=human_upright_v2
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/human_upright_v2_%j.out
#SBATCH --error=logs/human_upright_v2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# Strategy: Late DINO warmup (200 epochs) + lower reg + upright prompt
# Let shape become bipedal first, THEN apply DINO to preserve that

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/human_upright_v2 \
  --text_prompt "a standing person" \
  --use_dino_loss \
  --dino_weight 0.15 \
  --dino_warmup_epochs 200 \
  --regularize_jacobians_weight 20000 \
  --epochs 3300
