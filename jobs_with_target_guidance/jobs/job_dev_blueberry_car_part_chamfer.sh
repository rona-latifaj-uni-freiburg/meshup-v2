#!/bin/bash
#SBATCH --job-name=dev_car_pchamfer
#SBATCH --partition=dev_gpu_h100
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/dev_car_pchamfer_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/logs/dev_car_pchamfer_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

PROMPTS=(
  "a sports car"
  "an SUV"
)

TARGETS=(
  "./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply"
  "./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply"
)

LABELS=(
  "blueberry_to_bugatti_sportscar"
  "blueberry_to_santafe_suv"
)

PART_CHAMFER_WEIGHTS=(8000 7000)
GLOBAL_CHAMFER_WEIGHTS=(1000 1000)
TARGET_WEIGHTS=(150 130)
TARGET_RENDER_WEIGHTS=(10 8)
REG_WEIGHTS=(1800 2200)

PROMPT=${PROMPTS[$TASK_ID]}
TARGET=${TARGETS[$TASK_ID]}
LABEL=${LABELS[$TASK_ID]}
PART_CHAMFER_WEIGHT=${PART_CHAMFER_WEIGHTS[$TASK_ID]}
GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHTS[$TASK_ID]}
TARGET_WEIGHT=${TARGET_WEIGHTS[$TASK_ID]}
TARGET_RENDER_WEIGHT=${TARGET_RENDER_WEIGHTS[$TASK_ID]}
REG_WEIGHT=${REG_WEIGHTS[$TASK_ID]}

OUT=./jobs_with_target_guidance/outputs/${LABEL}_part_chamfer_${SLURM_JOB_ID:-manual}_${TASK_ID}

echo "======================================================"
echo "DEV H100 car part-aware target Chamfer"
echo "TASK_ID=${TASK_ID}"
echo "PROMPT=${PROMPT}"
echo "TARGET=${TARGET}"
echo "OUT=${OUT}"
echo "PART_CHAMFER_WEIGHT=${PART_CHAMFER_WEIGHT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "TARGET_WEIGHT=${TARGET_WEIGHT}"
echo "TARGET_RENDER_WEIGHT=${TARGET_RENDER_WEIGHT}"
echo "START_TIME=$(date)"
echo "======================================================"

python main.py \
  --config ./jobs_with_target_guidance/configs/car_part_chamfer.yml \
  --mesh ./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply \
  --text_prompt "${PROMPT}" \
  --target_mesh "${TARGET}" \
  --target_mesh_weight "${TARGET_WEIGHT}" \
  --target_mesh_render_weight "${TARGET_RENDER_WEIGHT}" \
  --target_mesh_chamfer_weight "${GLOBAL_CHAMFER_WEIGHT}" \
  --target_mesh_part_chamfer_weight "${PART_CHAMFER_WEIGHT}" \
  --regularize_jacobians_weight "${REG_WEIGHT}" \
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

