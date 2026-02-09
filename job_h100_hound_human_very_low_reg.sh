#!/bin/bash
#SBATCH --job-name=h100_hound_human_very_low_reg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_hound_human_low_reg_%j.out
#SBATCH --error=logs/h100_hound_human_low_reg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# VERY LOW REG: Hound → Human
# Minimal regularization to allow maximum pose change (quadruped→biped)

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/h100_hound_human_very_low_reg \
  --text_prompt "a human person standing upright on two legs, arms raised" \
  --use_dino_loss \
  --dino_weight 0.05 \
  --regularize_jacobians_weight 10000 \
  --epochs 15000
