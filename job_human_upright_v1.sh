#!/bin/bash
#SBATCH --job-name=human_upright_v1
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/human_upright_v1_%j.out
#SBATCH --error=logs/human_upright_v1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# Strategy: Explicit upright prompt + DINO
# The prompt explicitly describes bipedal stance to fight the quadruped pose

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/human_upright_v1 \
  --text_prompt "a human standing upright on two legs" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --epochs 3300
