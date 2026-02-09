#!/bin/bash
#SBATCH --job-name=h100_hound_to_human_long
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_hound_to_human_long_%j.out
#SBATCH --error=logs/h100_hound_to_human_long_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# EXTENDED: Hound → Human (2 hours, 15000 epochs)
# Quadruped to biped - the classic challenge
# Front paws→hands, back paws→feet, snout→nose, tail→shrink

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/h100_hound_to_human_long \
  --text_prompt "a human person standing upright on two legs, arms at sides" \
  --use_dino_loss \
  --dino_weight 0.08 \
  --regularize_jacobians_weight 20000 \
  --epochs 15000
