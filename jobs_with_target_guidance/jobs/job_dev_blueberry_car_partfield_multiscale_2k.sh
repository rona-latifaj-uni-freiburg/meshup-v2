#!/bin/bash
#SBATCH --job-name=dev_blue_pfms2k
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=jobs_with_target_guidance/logs/dev_blue_pfms2k_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/logs/dev_blue_pfms2k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

SOURCE=./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply
SOURCE_NAME=blueberry

if [[ "${TASK_ID}" -eq 0 ]]; then
  TARGET=./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply
  TARGET_NAME=bugatti
  LABEL=blueberry_to_bugatti_multiscale_2k_dev
  PROMPT="a sports car"
elif [[ "${TASK_ID}" -eq 1 ]]; then
  TARGET=./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply
  TARGET_NAME=santa_fe
  LABEL=blueberry_to_santafe_multiscale_2k_dev
  PROMPT="an SUV"
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
EPOCHS=2000
BLEND_EPOCHS=150

MULTISCALE_BUCKETS=(8 12 20)
MULTISCALE_LABEL_DIRS=(
  ./jobs_with_target_guidance/partfield_segments/car_5k_8
  ./jobs_with_target_guidance/partfield_segments/car_5k
  ./jobs_with_target_guidance/partfield_segments/car_5k_20
)

for label_dir in "${MULTISCALE_LABEL_DIRS[@]}"; do
  source_labels="${label_dir}/labels/${SOURCE_NAME}_partfield_labels.npz"
  target_labels="${label_dir}/labels/${TARGET_NAME}_partfield_labels.npz"
  if [[ ! -f "${source_labels}" || ! -f "${target_labels}" ]]; then
    echo "Missing multi-scale PartField labels in ${label_dir}"
    echo "Expected source labels: ${source_labels}"
    echo "Expected target labels: ${target_labels}"
    echo "Generate all scales with:"
    echo "  sbatch --export=ALL,N_BUCKETS_LIST=8,12,20 jobs_with_target_guidance/jobs/job_dev_partfield_segment_car_features.sh"
    exit 1
  fi
done

JOB_STEM=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=./jobs_with_target_guidance/outputs/${LABEL}_partfield_chamfer_${JOB_STEM}_${TASK_ID}

echo "======================================================"
echo "DEV blueberry-source multi-scale PartField Chamfer, task ${TASK_ID}, ${EPOCHS} epochs"
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
