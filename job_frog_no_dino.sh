#!/bin/bash
#SBATCH --job-name=frog_no_dino
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/frog_no_dino_%j.out
#SBATCH --error=logs/frog_no_dino_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
   --mesh ./meshes/hound.obj \
   --output_path ./outputs/frog_no_dino \
   --text_prompt "a frog" \
   --epochs 2500
