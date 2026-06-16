#!/bin/bash
#SBATCH --job-name=dev_newcar_pfms2k
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=jobs_with_target_guidance/logs/dev_newcar_pfms2k_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/logs/dev_newcar_pfms2k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MESH_DIR=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down

if [[ "${TASK_ID}" -eq 0 ]]; then
  SOURCE="${MESH_DIR}/f1_car_5k_upright_wheels_down.ply"
  TARGET="${MESH_DIR}/f1_verstappen_5k_upright_wheels_down.ply"
  SOURCE_NAME=f1_car
  TARGET_NAME=f1_verstappen
  LABEL=f1_car_to_f1_verstappen_multiscale_2k_dev
  PROMPT="a Formula 1 race car"
elif [[ "${TASK_ID}" -eq 1 ]]; then
  SOURCE="${MESH_DIR}/g_class_5k_upright_wheels_down.ply"
  TARGET="${MESH_DIR}/green_suv_5k_upright_wheels_down.ply"
  SOURCE_NAME=g_class
  TARGET_NAME=green_suv
  LABEL=g_class_to_green_suv_multiscale_2k_dev
  PROMPT="a green SUV"
else
  echo "Unknown TASK_ID=${TASK_ID}"
  exit 1
fi

PARTFIELD_CHAMFER_WEIGHT=8000
GLOBAL_CHAMFER_WEIGHT=750
TARGET_WEIGHT=130
TARGET_RENDER_WEIGHT=8
REG_WEIGHT=1800
POSITION_WEIGHT=0.05
EPOCHS=${EPOCHS:-2000}
BLEND_EPOCHS=150

MULTISCALE_BUCKETS=(8 12 20)
MULTISCALE_LABEL_DIRS=(
  ./jobs_with_target_guidance/partfield_segments/new_car_meshes_5k_8
  ./jobs_with_target_guidance/partfield_segments/new_car_meshes_5k
  ./jobs_with_target_guidance/partfield_segments/new_car_meshes_5k_20
)

for label_dir in "${MULTISCALE_LABEL_DIRS[@]}"; do
  source_labels="${label_dir}/labels/${SOURCE_NAME}_partfield_labels.npz"
  target_labels="${label_dir}/labels/${TARGET_NAME}_partfield_labels.npz"
  if [[ ! -f "${source_labels}" || ! -f "${target_labels}" ]]; then
    echo "Missing multi-scale PartField labels in ${label_dir}"
    echo "Expected source labels: ${source_labels}"
    echo "Expected target labels: ${target_labels}"
    echo "Generate them with:"
    echo "  sbatch jobs_with_target_guidance/jobs/job_prepare_partfield_new_car_meshes_features.sh"
    exit 1
  fi
done

JOB_STEM=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=./jobs_with_target_guidance/outputs/${LABEL}_partfield_chamfer_${JOB_STEM}_${TASK_ID}

echo "======================================================"
echo "DEV new-car multi-scale PartField Chamfer, task ${TASK_ID}, ${EPOCHS} epochs"
echo "SOURCE=${SOURCE}"
echo "PROMPT=${PROMPT}"
echo "TARGET=${TARGET}"
echo "MULTISCALE_BUCKETS=${MULTISCALE_BUCKETS[*]}"
echo "MULTISCALE_LABEL_DIRS=${MULTISCALE_LABEL_DIRS[*]}"
echo "OUT=${OUT}"
echo "PARTFIELD_CHAMFER_WEIGHT=${PARTFIELD_CHAMFER_WEIGHT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "TARGET_WEIGHT=${TARGET_WEIGHT}"
echo "TARGET_RENDER_WEIGHT=${TARGET_RENDER_WEIGHT}"
echo "BLEND_EPOCHS=${BLEND_EPOCHS}"
echo "START_TIME=$(date)"
echo "======================================================"

python main.py \
  --config ./jobs_with_target_guidance/configs/car_partfield_chamfer.yml \
  --mesh "${SOURCE}" \
  --text_prompt "${PROMPT}" \
  --target_mesh "${TARGET}" \
  --target_mesh_weight "${TARGET_WEIGHT}" \
  --target_mesh_render_weight "${TARGET_RENDER_WEIGHT}" \
  --target_mesh_chamfer_weight "${GLOBAL_CHAMFER_WEIGHT}" \
  --target_mesh_partfield_chamfer_weight "${PARTFIELD_CHAMFER_WEIGHT}" \
  --partfield_source_labels "${MULTISCALE_LABEL_DIRS[1]}/labels/${SOURCE_NAME}_partfield_labels.npz" \
  --partfield_target_labels "${MULTISCALE_LABEL_DIRS[1]}/labels/${TARGET_NAME}_partfield_labels.npz" \
  --partfield_labels_aligned \
  --partfield_label_mode auto \
  --partfield_position_weight "${POSITION_WEIGHT}" \
  --partfield_multiscale_enabled \
  --partfield_multiscale_buckets "${MULTISCALE_BUCKETS[@]}" \
  --partfield_multiscale_label_dirs "${MULTISCALE_LABEL_DIRS[@]}" \
  --partfield_multiscale_blend_epochs "${BLEND_EPOCHS}" \
  --regularize_jacobians_weight "${REG_WEIGHT}" \
  --output_path "${OUT}" \
  --epochs "${EPOCHS}"

python generate_pca_evolution_4views.py \
  --epoch_renders_dir "${OUT}/epoch_renders" \
  --output_dir "${OUT}/pca_initial_final" \
  --epochs "1,${EPOCHS}" \
  --model dinov2_vitl14 \
  --image_size 518

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
