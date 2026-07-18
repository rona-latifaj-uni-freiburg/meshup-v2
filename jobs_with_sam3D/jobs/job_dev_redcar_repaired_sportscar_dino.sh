#!/bin/bash
#SBATCH --job-name=dev_redcar_sport_dino
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_sam3D/logs/dev_redcar_sport_dino_%j.out
#SBATCH --error=jobs_with_sam3D/logs/dev_redcar_sport_dino_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail
cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs
source ./activate_meshup_new.sh

python main.py \
  --config ./configs/base_config.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_repaired/red_car_5k_repaired.ply \
  --text_prompt "sports car" \
  --use_dino_loss \
  --dino_weight 0.1 \
  --dino_warmup_epochs 200 \
  --azim_min 300 \
  --azim_max 350 \
  --elev_max 35 \
  --output_path ./jobs_with_sam3D/outputs/redcar_repaired_to_sportscar_dino_${SLURM_JOB_ID:-manual} \
  --epochs 2500
