#!/bin/bash
#SBATCH --job-name=h100_hound_hippo_textreme
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=jobs_with_sam3D/logs/h100_hound_hippo_textreme_%j.out
#SBATCH --error=jobs_with_sam3D/logs/h100_hound_hippo_textreme_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

python main.py \
  --config ./configs/base_config_target_extreme.yml \
  --mesh ./meshes/hound.obj \
  --text_prompt "a hippopotamus with very big ears" \
  --no-use_dino_loss \
  --use_target_mesh_guidance \
  --target_mesh ./experiments/simple_meshes/outputs/hippo_bigears/mesh_final/mesh.obj \
  --target_mesh_weight 12.0 \
  --target_mesh_warmup_epochs 1 \
  --target_mesh_global_weight 2.0 \
  --target_mesh_spatial_weight 5.0 \
  --target_mesh_render_weight 0.5 \
  --target_mesh_n_azimuths 20 \
  --target_mesh_n_elevations 5 \
  --target_mesh_online_render \
  --target_mesh_online_cache \
  --target_mesh_online_cache_max 12000 \
  --target_mesh_view_rounding_deg 3 \
  --target_mesh_view_rounding_dist 0.05 \
  --target_mesh_view_rounding_fov 1 \
  --azim_min 0 \
  --azim_max 360 \
  --elev_max 60 \
  --epochs 4000 \
  --output_path ./jobs_with_sam3D/outputs/hound_to_hippo_bigears_target_extreme_${SLURM_JOB_ID:-manual}
