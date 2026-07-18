#!/bin/bash
#SBATCH --job-name=dev_bug_chlock
#SBATCH --partition=dev_gpu_h100
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_sam3D/logs/dev_bug_chlock_%A_%a.out
#SBATCH --error=jobs_with_sam3D/logs/dev_bug_chlock_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

PROMPTS=(
  "a sports car with a rear spoiler"
  "a bugatti centodieci"
  "a bugatti centodieci with a rear spoiler"
  "a bugatti centodieci with a large rear spoiler"
)

LABELS=(
  "prompt_sportscar_rear_spoiler"
  "prompt_bugatti_centodieci"
  "prompt_bugatti_centodieci_rear_spoiler"
  "prompt_bugatti_centodieci_large_rear_spoiler"
)

CHAMFER_WEIGHTS=(8000 10000 12000 16000)
TARGET_WEIGHTS=(160 180 220 260)
RENDER_WEIGHTS=(10 12 15 18)
IMAGE_WEIGHTS=(1.5 1.0 0.75 0.35)
IMAGE_START_FACTORS=(0.08 0.05 0.03 0.02)
SDS_STARTS=(0.05 0.04 0.03 0.02)
SDS_MIDS=(0.25 0.20 0.15 0.10)
TARGET_FLOORS=(0.95 0.98 1.00 1.00)
REG_WEIGHTS=(2500 2200 1800 1400)

PROMPT=${PROMPTS[$TASK_ID]}
LABEL=${LABELS[$TASK_ID]}
CHAMFER_WEIGHT=${CHAMFER_WEIGHTS[$TASK_ID]}
TARGET_WEIGHT=${TARGET_WEIGHTS[$TASK_ID]}
RENDER_WEIGHT=${RENDER_WEIGHTS[$TASK_ID]}
IMAGE_WEIGHT=${IMAGE_WEIGHTS[$TASK_ID]}
IMAGE_START_FACTOR=${IMAGE_START_FACTORS[$TASK_ID]}
SDS_START=${SDS_STARTS[$TASK_ID]}
SDS_MID=${SDS_MIDS[$TASK_ID]}
TARGET_FLOOR=${TARGET_FLOORS[$TASK_ID]}
REG_WEIGHT=${REG_WEIGHTS[$TASK_ID]}

OUT=./jobs_with_sam3D/outputs/blueberry_to_bugatti_${LABEL}_target_chamfer_lock_3000_${SLURM_JOB_ID:-manual}_${TASK_ID}

echo "======================================================"
echo "DEV H100 Bugatti Chamfer-lock run"
echo "TASK_ID=${TASK_ID}"
echo "PROMPT=${PROMPT}"
echo "OUT=${OUT}"
echo "CHAMFER_WEIGHT=${CHAMFER_WEIGHT}"
echo "TARGET_WEIGHT=${TARGET_WEIGHT}"
echo "RENDER_WEIGHT=${RENDER_WEIGHT}"
echo "IMAGE_WEIGHT=${IMAGE_WEIGHT}"
echo "START_TIME=$(date)"
echo "======================================================"

python main.py \
  --config ./configs/base_config_target_dominant.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply \
  --text_prompt "${PROMPT}" \
  --guidance_scale 7 \
  --image_weight "${IMAGE_WEIGHT}" \
  --image_weight_start_factor "${IMAGE_START_FACTOR}" \
  --image_weight_ramp_epochs 900 \
  --loss_schedule three_stage \
  --loss_schedule_stage1_epochs 900 \
  --loss_schedule_stage2_epochs 2100 \
  --loss_schedule_sds_start "${SDS_START}" \
  --loss_schedule_sds_mid "${SDS_MID}" \
  --loss_schedule_target_floor "${TARGET_FLOOR}" \
  --regularize_jacobians_weight "${REG_WEIGHT}" \
  --no-use_dino_loss \
  --use_target_mesh_guidance \
  --target_mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply \
  --target_mesh_weight "${TARGET_WEIGHT}" \
  --target_mesh_warmup_epochs 15 \
  --target_mesh_global_weight 1.0 \
  --target_mesh_spatial_weight 4.0 \
  --target_mesh_render_weight "${RENDER_WEIGHT}" \
  --target_mesh_chamfer_weight "${CHAMFER_WEIGHT}" \
  --target_mesh_chamfer_warmup_epochs 20 \
  --target_mesh_chamfer_points 4096 \
  --target_mesh_n_azimuths 16 \
  --target_mesh_n_elevations 4 \
  --target_mesh_online_render \
  --target_mesh_online_cache \
  --target_mesh_online_cache_max 4096 \
  --target_mesh_view_rounding_deg 5 \
  --target_mesh_view_rounding_dist 0.1 \
  --target_mesh_view_rounding_fov 2 \
  --save_epoch_renders \
  --save_renders_interval 300 \
  --epoch_render_res 512 \
  --log_interval_im 300 \
  --azim_min 0 \
  --azim_max 360 \
  --elev_max 60 \
  --output_path "${OUT}" \
  --epochs 3000

python generate_pca_evolution_4views.py \
  --epoch_renders_dir "${OUT}/epoch_renders" \
  --output_dir "${OUT}/pca_initial_final" \
  --epochs "1,3000" \
  --model dinov2_vitl14 \
  --image_size 518

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
