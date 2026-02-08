#!/bin/bash
#SBATCH --job-name=cat_with_dino
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cat_with_dino_%j.out
#SBATCH --error=logs/cat_with_dino_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
   --mesh ./meshes/hound.obj \
   --output_path ./outputs/cat_with_dino \
   --text_prompt "a cat" \
   --use_dino_loss \
   --epochs 2000
