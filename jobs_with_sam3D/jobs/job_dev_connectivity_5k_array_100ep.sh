#!/bin/bash
#SBATCH --job-name=dev_conn5k_100ep
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --array=0-9
#SBATCH --output=jobs_with_sam3D/logs/dev_conn5k_100ep_%A_%a.out
#SBATCH --error=jobs_with_sam3D/logs/dev_conn5k_100ep_%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/outputs jobs_with_sam3D/slurm_logs

source ./activate_meshup_new.sh

meshes=(
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/kona_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/old_car_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/oldie_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/passati_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/red_car_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/usa_suv_5k_upright_wheels_down.ply
  ./jobs_with_sam3D/meshes/5k_upright_wheels_down/vintage_car_5k_upright_wheels_down.ply
)

mesh_path="${meshes[$SLURM_ARRAY_TASK_ID]}"
mesh_name="$(basename "$mesh_path" .ply)"

echo "======================================================"
echo "DEV CONNECTIVITY CHECK: $mesh_name"
echo "======================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "HOST=$(hostname)"
echo "START_TIME=$(date)"
echo "MESH=$mesh_path"

python main.py \
  --config ./configs/base_config.yml \
  --mesh "$mesh_path" \
  --text_prompt "car" \
  --azim_min 300 \
  --azim_max 350 \
  --elev_max 35 \
  --output_path "./jobs_with_sam3D/outputs/connectivity_100ep_${mesh_name}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}" \
  --epochs 100

echo "END_TIME=$(date)"
echo "Finished connectivity check for $mesh_name"
