#!/bin/bash
#SBATCH --job-name=human_strong_dino
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/human_strong_dino_low_reg_%j.out
#SBATCH --error=logs/human_strong_dino_low_reg_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/human_strong_dino_low_reg.yml
