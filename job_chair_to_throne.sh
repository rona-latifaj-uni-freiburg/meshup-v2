#!/bin/bash
#SBATCH --job-name=chair_throne
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/chair_to_throne_%j.out
#SBATCH --error=logs/chair_to_throne_%j.err

# Chair → Throne
# Furniture transformation - tests structural correspondence

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-chair_001/Aligned.obj \
    --text_prompt "an ornate royal throne, golden, with high back and armrests" \
    --output_path outputs/chair_to_throne_tracked \
    --epochs 3000 \
    --reg_weight 50000 \
    --use_dino_loss \
    --dino_loss_weight 0.1 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Chair to throne job complete!"
