#!/bin/bash
#SBATCH --job-name=dev_bug_tgtdom
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_sam3D/logs/dev_bug_tgtdom_%j.out
#SBATCH --error=jobs_with_sam3D/logs/dev_bug_tgtdom_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

PROMPT="a sports car with a rear spoiler"
OUT=./jobs_with_sam3D/outputs/blueberry_to_bugatti_prompt_sportscar_rear_spoiler_target_dominant_2500_${SLURM_JOB_ID:-manual}

echo "======================================================"
echo "DEV H100 Bugatti spoiler target-dominant run"
echo "PROMPT=${PROMPT}"
echo "OUT=${OUT}"
echo "START_TIME=$(date)"
echo "======================================================"

python main.py \
  --config ./configs/base_config_target_dominant.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply \
  --text_prompt "${PROMPT}" \
  --guidance_scale 7 \
  --image_weight 2.0 \
  --image_weight_start_factor 0.05 \
  --image_weight_ramp_epochs 900 \
  --loss_schedule three_stage \
  --loss_schedule_stage1_epochs 800 \
  --loss_schedule_stage2_epochs 1800 \
  --loss_schedule_sds_start 0.03 \
  --loss_schedule_sds_mid 0.25 \
  --loss_schedule_target_floor 1.00 \
  --regularize_jacobians_weight 3000 \
  --no-use_dino_loss \
  --use_target_mesh_guidance \
  --target_mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply \
  --target_mesh_weight 160.0 \
  --target_mesh_warmup_epochs 20 \
  --target_mesh_global_weight 1.0 \
  --target_mesh_spatial_weight 3.5 \
  --target_mesh_render_weight 10.0 \
  --target_mesh_chamfer_weight 6000 \
  --target_mesh_chamfer_warmup_epochs 30 \
  --target_mesh_chamfer_points 2048 \
  --target_mesh_n_azimuths 12 \
  --target_mesh_n_elevations 3 \
  --target_mesh_online_render \
  --target_mesh_online_cache \
  --target_mesh_online_cache_max 4096 \
  --target_mesh_view_rounding_deg 5 \
  --target_mesh_view_rounding_dist 0.1 \
  --target_mesh_view_rounding_fov 2 \
  --save_epoch_renders \
  --save_renders_interval 250 \
  --epoch_render_res 512 \
  --log_interval_im 250 \
  --azim_min 0 \
  --azim_max 360 \
  --elev_max 60 \
  --output_path "${OUT}" \
  --epochs 2500

python generate_pca_evolution_4views.py \
  --epoch_renders_dir "${OUT}/epoch_renders" \
  --output_dir "${OUT}/pca_initial_final" \
  --epochs "1,2500" \
  --model dinov2_vitl14 \
  --image_size 518

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
