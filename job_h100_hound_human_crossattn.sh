#!/bin/bash
#SBATCH --job-name=h100_hound_human_crossattn
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_hound_human_crossattn_%j.out
#SBATCH --error=logs/h100_hound_human_crossattn_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# CROSS-ATTENTION GUIDANCE: Hound → Human
# Uses diffusion cross-attention maps for semantic guidance

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/h100_hound_human_crossattn \
  --text_prompt "a human person standing upright on two legs" \
  --use_cross_attn_loss \
  --cross_attn_weight 0.1 \
  --regularize_jacobians_weight 25000 \
  --epochs 12000
