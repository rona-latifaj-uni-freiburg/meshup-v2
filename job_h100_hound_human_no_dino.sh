#!/bin/bash
#SBATCH --job-name=h100_hound_human_no_dino
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_hound_human_no_dino_%j.out
#SBATCH --error=logs/h100_hound_human_no_dino_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# BASELINE (NO DINO): Hound → Human
# Control experiment - what happens without DINO?

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/h100_hound_human_no_dino \
  --text_prompt "a human person standing upright on two legs" \
  --regularize_jacobians_weight 25000 \
  --epochs 15000
