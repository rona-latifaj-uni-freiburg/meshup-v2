#!/bin/bash
#SBATCH --job-name=h100_hound_human_both_losses
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=logs/h100_hound_human_both_%j.out
#SBATCH --error=logs/h100_hound_human_both_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# BOTH DINO + CROSS-ATTENTION: Hound → Human
# Combining both semantic guidance methods!

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/h100_hound_human_both_losses \
  --text_prompt "a human person standing upright on two legs, T-pose" \
  --use_dino_loss \
  --dino_weight 0.08 \
  --use_cross_attn_loss \
  --cross_attn_weight 0.08 \
  --regularize_jacobians_weight 20000 \
  --epochs 15000
