#!/bin/bash
#SBATCH --job-name=starfish_octopus
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/starfish_octopus_%j.out
#SBATCH --error=logs/starfish_octopus_%j.err

# Starfish → Octopus
# Radial symmetry test - 5 arms to 8 tentacles

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-starfish_001/Aligned.obj \
    --text_prompt "an octopus, purple, with curling tentacles, underwater creature" \
    --output_path outputs/starfish_to_octopus_tracked \
    --epochs 3500 \
    --reg_weight 30000 \
    --use_dino_loss \
    --dino_loss_weight 0.08 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Starfish to octopus job complete!"
