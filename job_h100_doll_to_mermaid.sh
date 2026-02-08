#!/bin/bash
#SBATCH --job-name=h100_doll_to_mermaid
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:50:00
#SBATCH --output=logs/h100_doll_to_mermaid_%j.out
#SBATCH --error=logs/h100_doll_to_mermaid_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=latifajrona@gmail.com

# NO TAIL → WITH TAIL: Doll (legs) → Mermaid (fish tail)
# Tests: Can legs fuse into a tail? Upper body preserved?

mkdir -p logs
source ./activate_meshup_new.sh

python main.py --config ./configs/tracked_config.yml \
  --mesh ./data/Omni6DPose/PAM/object_meshes/omniobject3d-doll_001/Aligned.obj \
  --output_path ./outputs/h100_doll_to_mermaid \
  --text_prompt "a mermaid with fish tail" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --regularize_jacobians_weight 30000 \
  --epochs 5000
