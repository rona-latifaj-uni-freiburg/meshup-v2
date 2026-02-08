#!/bin/bash
#SBATCH --job-name=kiwi_no_dino
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/kiwi_no_dino_%j.out
#SBATCH --error=logs/kiwi_no_dino_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment
source ./activate_meshup_new.sh

# Run the experiment
python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/kiwi_no_dino \
  --text_prompt "a kiwi bird" \
  --epochs 2500
