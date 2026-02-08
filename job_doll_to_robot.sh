#!/bin/bash
#SBATCH --job-name=doll_robot
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/doll_to_robot_%j.out
#SBATCH --error=logs/doll_to_robot_%j.err

# Doll → Robot
# Humanoid to humanoid - great for testing limb correspondence

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-doll_001/Aligned.obj \
    --text_prompt "a humanoid robot, metallic, standing pose, sci-fi android" \
    --output_path outputs/doll_to_robot_tracked \
    --epochs 3500 \
    --reg_weight 40000 \
    --use_dino_loss \
    --dino_loss_weight 0.1 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Doll to robot job complete!"
