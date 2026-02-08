#!/bin/bash
#SBATCH --job-name=teapot_to_can
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/teapot_to_can_%j.out
#SBATCH --error=logs/teapot_to_can_%j.err

# Teapot → Watering can
# Classic shape morph - tests handle/spout correspondence

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-teapot_001/Aligned.obj \
    --text_prompt "a garden watering can, metal, with long spout" \
    --output_path outputs/teapot_to_watering_can_tracked \
    --epochs 3000 \
    --reg_weight 50000 \
    --use_dino_loss \
    --dino_loss_weight 0.1 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Teapot to watering can job complete!"
