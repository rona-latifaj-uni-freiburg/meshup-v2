#!/bin/bash
#SBATCH --job-name=dev_newcar_tg3k
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=jobs_with_target_guidance/logs/dev_newcar_tg3k_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/logs/dev_newcar_tg3k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs_new_cars jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
MESH_DIR=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down

if [[ "${TASK_ID}" -eq 0 ]]; then
  SOURCE="${MESH_DIR}/f1_car_5k_upright_wheels_down.ply"
  TARGET="${MESH_DIR}/f1_verstappen_5k_upright_wheels_down.ply"
  SOURCE_NAME=f1_car
  TARGET_NAME=f1_verstappen
  SOURCE_FEATURE_NAME=f1_car_5k_upright_wheels_down
  TARGET_FEATURE_NAME=f1_verstappen_5k_upright_wheels_down
  LABEL=f1_car_to_f1_verstappen_general_3k_dev
  PROMPT="an f1 car"
elif [[ "${TASK_ID}" -eq 1 ]]; then
  SOURCE="${MESH_DIR}/f1_verstappen_5k_upright_wheels_down.ply"
  TARGET="${MESH_DIR}/f1_car_5k_upright_wheels_down.ply"
  SOURCE_NAME=f1_verstappen
  TARGET_NAME=f1_car
  SOURCE_FEATURE_NAME=f1_verstappen_5k_upright_wheels_down
  TARGET_FEATURE_NAME=f1_car_5k_upright_wheels_down
  LABEL=f1_verstappen_to_f1_car_general_3k_dev
  PROMPT="an f1 car"
elif [[ "${TASK_ID}" -eq 2 ]]; then
  SOURCE="${MESH_DIR}/red_truck1_5k_upright_wheels_down.ply"
  TARGET="${MESH_DIR}/red_truck2_5k_upright_wheels_down.ply"
  SOURCE_NAME=red_truck1
  TARGET_NAME=red_truck2
  SOURCE_FEATURE_NAME=red_truck1_5k_upright_wheels_down
  TARGET_FEATURE_NAME=red_truck2_5k_upright_wheels_down
  LABEL=red_truck1_to_red_truck2_general_3k_dev
  PROMPT="a truck car"
elif [[ "${TASK_ID}" -eq 3 ]]; then
  SOURCE="${MESH_DIR}/mini_cooper_5k_upright_wheels_down.ply"
  TARGET="${MESH_DIR}/g_class_5k_upright_wheels_down.ply"
  SOURCE_NAME=mini_cooper
  TARGET_NAME=g_class
  SOURCE_FEATURE_NAME=mini_cooper_5k_upright_wheels_down
  TARGET_FEATURE_NAME=g_class_5k_upright_wheels_down
  LABEL=mini_cooper_to_g_class_suv_3k_dev
  PROMPT="an SUV"
else
  echo "Unknown TASK_ID=${TASK_ID}"
  exit 1
fi

PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/new_car_target_runs_5k
SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_FEATURE_NAME}_0_batch.npy
TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_FEATURE_NAME}_0_batch.npy

if [[ ! -f "${SOURCE_PARTFIELD_FEATURES}" ]]; then
  SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_FEATURE_NAME}_0.npy
fi
if [[ ! -f "${TARGET_PARTFIELD_FEATURES}" ]]; then
  TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_FEATURE_NAME}_0.npy
fi

MULTISCALE_BUCKETS=(8 12 20)
MULTISCALE_LABEL_DIRS=(
  ./jobs_with_target_guidance/partfield_segments/new_car_target_runs_5k_8
  ./jobs_with_target_guidance/partfield_segments/new_car_target_runs_5k
  ./jobs_with_target_guidance/partfield_segments/new_car_target_runs_5k_20
)

for path in "${SOURCE_PARTFIELD_FEATURES}" "${TARGET_PARTFIELD_FEATURES}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required PartField feature input: ${path}"
    echo "Generate features with:"
    echo "  sbatch jobs_with_target_guidance/jobs/job_prepare_partfield_new_car_target_runs_features.sh"
    exit 1
  fi
done

for label_dir in "${MULTISCALE_LABEL_DIRS[@]}"; do
  source_labels="${label_dir}/labels/${SOURCE_NAME}_partfield_labels.npz"
  target_labels="${label_dir}/labels/${TARGET_NAME}_partfield_labels.npz"
  if [[ ! -f "${source_labels}" || ! -f "${target_labels}" ]]; then
    echo "Missing multi-scale PartField labels in ${label_dir}"
    echo "Expected source labels: ${source_labels}"
    echo "Expected target labels: ${target_labels}"
    echo "Generate labels with:"
    echo "  sbatch jobs_with_target_guidance/jobs/job_prepare_partfield_new_car_target_runs_features.sh"
    exit 1
  fi
done

PARTFIELD_CHAMFER_WEIGHT=8000
GLOBAL_CHAMFER_WEIGHT=750
TARGET_WEIGHT=130
TARGET_RENDER_WEIGHT=8
REG_WEIGHT=1800
EPOCHS=${EPOCHS:-3000}
BLEND_EPOCHS=150
SOFT_POINTS=1024
SOFT_SEMANTIC_WEIGHT=0.10
SOFT_TEMPERATURE=0.03
HARD_WEIGHT=0.60
SOFT_WEIGHT=0.40

JOB_STEM=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=./jobs_with_target_guidance/outputs_new_cars/${LABEL}_partfield_hybrid_${JOB_STEM}_${TASK_ID}

echo "======================================================"
echo "DEV new-car target guidance, task ${TASK_ID}, ${EPOCHS} epochs"
echo "SOURCE=${SOURCE}"
echo "PROMPT=${PROMPT}"
echo "TARGET=${TARGET}"
echo "SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES}"
echo "TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES}"
echo "MULTISCALE_BUCKETS=${MULTISCALE_BUCKETS[*]}"
echo "MULTISCALE_LABEL_DIRS=${MULTISCALE_LABEL_DIRS[*]}"
echo "OUT=${OUT}"
echo "PARTFIELD_CHAMFER_WEIGHT=${PARTFIELD_CHAMFER_WEIGHT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "TARGET_WEIGHT=${TARGET_WEIGHT}"
echo "TARGET_RENDER_WEIGHT=${TARGET_RENDER_WEIGHT}"
echo "HARD_WEIGHT=${HARD_WEIGHT}"
echo "SOFT_WEIGHT=${SOFT_WEIGHT}"
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
  --partfield_source_features "${SOURCE_PARTFIELD_FEATURES}" \
  --partfield_target_features "${TARGET_PARTFIELD_FEATURES}" \
  --partfield_source_labels "${MULTISCALE_LABEL_DIRS[1]}/labels/${SOURCE_NAME}_partfield_labels.npz" \
  --partfield_target_labels "${MULTISCALE_LABEL_DIRS[1]}/labels/${TARGET_NAME}_partfield_labels.npz" \
  --partfield_labels_aligned \
  --partfield_label_mode auto \
  --partfield_feature_mode auto \
  --partfield_multiscale_enabled \
  --partfield_multiscale_buckets "${MULTISCALE_BUCKETS[@]}" \
  --partfield_multiscale_label_dirs "${MULTISCALE_LABEL_DIRS[@]}" \
  --partfield_multiscale_blend_epochs "${BLEND_EPOCHS}" \
  --partfield_guidance_mode hybrid \
  --partfield_hard_weight "${HARD_WEIGHT}" \
  --partfield_soft_weight "${SOFT_WEIGHT}" \
  --partfield_soft_points "${SOFT_POINTS}" \
  --partfield_soft_semantic_weight "${SOFT_SEMANTIC_WEIGHT}" \
  --partfield_soft_temperature "${SOFT_TEMPERATURE}" \
  --regularize_jacobians_weight "${REG_WEIGHT}" \
  --output_path "${OUT}" \
  --epochs "${EPOCHS}"

python jobs_with_target_guidance/evaluate_target_pipeline.py \
  --output-dir "${OUT}" \
  --samples 3000 \
  --part-samples 750

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
