#!/bin/bash
#SBATCH --job-name=teddy_to_bear
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32gb
#SBATCH --output=logs/teddy_to_bear_%j.out
#SBATCH --error=logs/teddy_to_bear_%j.err

# Teddy bear toy → realistic grizzly bear
# Good test for soft toy to real animal correspondence

source activate_meshup_new.sh

python main.py \
    --mesh_path ./data/Omni6DPose/PAM/object_meshes/omniobject3d-teddy_bear_001/Aligned.obj \
    --text_prompt "a realistic grizzly bear, sitting, brown fur, detailed" \
    --output_path outputs/teddy_to_grizzly_tracked \
    --epochs 3500 \
    --reg_weight 50000 \
    --use_dino_loss \
    --dino_loss_weight 0.1 \
    --dino_warmup_epochs 50 \
    --enable_vertex_color_tracking

echo "Teddy to grizzly bear job complete!"
