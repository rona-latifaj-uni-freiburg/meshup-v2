#!/bin/bash
#SBATCH --job-name=plane_to_jet
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/plane_to_jet_%j.out
#SBATCH --error=logs/plane_to_jet_%j.err

# Toy plane → Fighter jet
# Aviation transformation - wing/fuselage correspondence

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_plane_001/Aligned.obj \
    --text_prompt "a stealth fighter jet, F-22 raptor style, military aircraft" \
    --output_path outputs/toy_plane_to_jet_tracked \
    --epochs 3000 \
    --reg_weight 50000 \
    --use_dino_loss \
    --dino_loss_weight 0.1 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Toy plane to fighter jet job complete!"
