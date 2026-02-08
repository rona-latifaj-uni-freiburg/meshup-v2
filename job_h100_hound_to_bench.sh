#!/bin/bash
#SBATCH --job-name=h100_hound_to_bench
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_hound_to_bench_%j.out
#SBATCH --error=logs/h100_hound_to_bench_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# EXTREME CROSS-CATEGORY: Hound dog → Park bench
# Tests: Can legs become bench legs? Body becomes seat?

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/h100_hound_to_bench \
  --text_prompt "a wooden park bench with backrest" \
  --use_dino_loss \
  --dino_weight 0.05 \
  --regularize_jacobians_weight 30000 \
  --epochs 5000
