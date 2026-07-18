#!/bin/bash
#SBATCH --job-name=h100_blue_bug_tgt_3st
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=jobs_with_sam3D/logs/h100_blue_bug_tgt_3st_%j.out
#SBATCH --error=jobs_with_sam3D/logs/h100_blue_bug_tgt_3st_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

python main.py \
  --config ./configs/base_config_target_dominant.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply \
  --text_prompt "a sports car" \
  --guidance_scale 8 \
  --image_weight 5.0 \
  --image_weight_start_factor 0.2 \
  --image_weight_ramp_epochs 1000 \
  --loss_schedule three_stage \
  --loss_schedule_stage1_epochs 900 \
  --loss_schedule_stage2_epochs 2400 \
  --loss_schedule_sds_start 0.10 \
  --loss_schedule_sds_mid 0.45 \
  --loss_schedule_target_floor 0.80 \
  --regularize_jacobians_weight 8000 \
  --no-use_dino_loss \
  --use_target_mesh_guidance \
  --target_mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply \
  --target_mesh_weight 80.0 \
  --target_mesh_warmup_epochs 40 \
  --target_mesh_global_weight 1.0 \
  --target_mesh_spatial_weight 2.5 \
  --target_mesh_render_weight 5.0 \
  --target_mesh_chamfer_weight 2500 \
  --target_mesh_chamfer_warmup_epochs 50 \
  --target_mesh_chamfer_points 3072 \
  --target_mesh_n_azimuths 16 \
  --target_mesh_n_elevations 4 \
  --target_mesh_online_render \
  --target_mesh_online_cache \
  --target_mesh_online_cache_max 8192 \
  --target_mesh_view_rounding_deg 5 \
  --target_mesh_view_rounding_dist 0.1 \
  --target_mesh_view_rounding_fov 2 \
  --azim_min 0 \
  --azim_max 360 \
  --elev_max 60 \
  --output_path ./jobs_with_sam3D/outputs/blueberry_to_bugatti_prompt_sportscar_target_three_stage_${SLURM_JOB_ID:-manual} \
  --epochs 3500
