#!/bin/bash
#SBATCH --job-name=human_dino_strong
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/human_dino_strong_%j.out
#SBATCH --error=logs/human_dino_strong_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
   --mesh ./meshes/hound.obj \
   --output_path ./outputs/human_dino_strong \
   --text_prompt "a human" \
   --use_dino_loss \
   --dino_weight 0.2 \
   --epochs 2500
