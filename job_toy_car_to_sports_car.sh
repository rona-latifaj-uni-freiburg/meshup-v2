#!/bin/bash
#SBATCH --job-name=car_transform
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/toy_car_to_sports_%j.out
#SBATCH --error=logs/toy_car_to_sports_%j.err

# Toy car → Sports car
# Vehicle transformation - wheels, body correspondence

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-toy_car_001/Aligned.obj \
    --text_prompt "a sleek red sports car, Ferrari style, realistic" \
    --output_path outputs/toy_car_to_sports_tracked \
    --epochs 3000 \
    --reg_weight 50000 \
    --use_dino_loss \
    --dino_loss_weight 0.1 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Toy car to sports car job complete!"
