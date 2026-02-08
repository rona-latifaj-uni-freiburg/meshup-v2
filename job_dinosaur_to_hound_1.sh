#!/bin/bash
#SBATCH --job-name=h100_dino_to_hound
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_dino_to_hound%j.out
#SBATCH --error=logs/h100_dino_to_hound%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# Transform dinosaur mesh into a hound with DINO loss for semantic correspondence

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-dinosaur_032/Aligned.obj \
  --output_path ./outputs/h100_dino_to_hound \
  --text_prompt "a hound dog" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --regularize_jacobians_weight 40000 \
  --epochs 5000
