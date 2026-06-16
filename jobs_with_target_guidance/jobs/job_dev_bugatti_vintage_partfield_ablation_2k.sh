#!/bin/bash
#SBATCH --job-name=dev_bug_vint_ab2k
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-7
#SBATCH --output=jobs_with_target_guidance/logs/dev_bug_vint_ab2k_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/logs/dev_bug_vint_ab2k_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DIRECTION_ID=$((TASK_ID / 4))
VARIANT_ID=$((TASK_ID % 4))

BUGATTI_MESH=./jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply
VINTAGE_MESH=./jobs_with_sam3D/meshes/5k_upright_wheels_down/vintage_car_5k_upright_wheels_down.ply

TARGET_WEIGHT=130
TARGET_RENDER_WEIGHT=8
REG_WEIGHT=1800
PARTFIELD_CHAMFER_WEIGHT=8000
GLOBAL_CHAMFER_WEIGHT=750
N_BUCKETS=12
POSITION_WEIGHT=0.05
EPOCHS=2000
VARIANT=baseline12
PARTFIELD_SEGMENT_DIR=./jobs_with_target_guidance/partfield_segments/bugatti_vintage_5k

case "${VARIANT_ID}" in
  0)
    VARIANT=baseline12
    ;;
  1)
    VARIANT=buckets20
    N_BUCKETS=20
    PARTFIELD_SEGMENT_DIR=./jobs_with_target_guidance/partfield_segments/bugatti_vintage_5k_20
    ;;
  2)
    VARIANT=soft_pf12
    PARTFIELD_CHAMFER_WEIGHT=4000
    ;;
  3)
    VARIANT=strong_global12
    GLOBAL_CHAMFER_WEIGHT=1500
    ;;
  *)
    echo "Unknown VARIANT_ID=${VARIANT_ID}"
    exit 1
    ;;
esac

PARTFIELD_LABEL_DIR=${PARTFIELD_SEGMENT_DIR}/labels
BUCKET_SUFFIX=$(printf "%02d" "${N_BUCKETS}")

if [[ "${DIRECTION_ID}" -eq 0 ]]; then
  SOURCE="${BUGATTI_MESH}"
  TARGET="${VINTAGE_MESH}"
  SOURCE_NAME=bugatti
  TARGET_NAME=vintage_car
  LABEL="bugatti_to_vintage_${VARIANT}_2k_dev"
  PROMPT="a vintage car"
else
  SOURCE="${VINTAGE_MESH}"
  TARGET="${BUGATTI_MESH}"
  SOURCE_NAME=vintage_car
  TARGET_NAME=bugatti
  LABEL="vintage_to_bugatti_${VARIANT}_2k_dev"
  PROMPT="a sports car"
fi

SOURCE_PARTFIELD_LABELS="${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz"
TARGET_PARTFIELD_LABELS="${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz"
SOURCE_PARTFIELD_COLORED="${PARTFIELD_SEGMENT_DIR}/colored/${SOURCE_NAME}_partfield_${BUCKET_SUFFIX}_parts.ply"
TARGET_PARTFIELD_COLORED="${PARTFIELD_SEGMENT_DIR}/colored/${TARGET_NAME}_partfield_${BUCKET_SUFFIX}_parts.ply"

if [[ ! -f "${SOURCE_PARTFIELD_LABELS}" || ! -f "${TARGET_PARTFIELD_LABELS}" ]]; then
  echo "Missing PartField label files for ${VARIANT}."
  echo "Expected source labels: ${SOURCE_PARTFIELD_LABELS}"
  echo "Expected target labels: ${TARGET_PARTFIELD_LABELS}"
  exit 1
fi

JOB_STEM=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=./jobs_with_target_guidance/outputs/${LABEL}_partfield_chamfer_${JOB_STEM}_${TASK_ID}

echo "======================================================"
echo "DEV Bugatti/Vintage PartField ablation, task ${TASK_ID}, ${EPOCHS} epochs"
echo "VARIANT=${VARIANT}"
echo "SOURCE=${SOURCE}"
echo "PROMPT=${PROMPT}"
echo "TARGET=${TARGET}"
echo "SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS}"
echo "TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS}"
echo "SOURCE_PARTFIELD_COLORED=${SOURCE_PARTFIELD_COLORED}"
echo "TARGET_PARTFIELD_COLORED=${TARGET_PARTFIELD_COLORED}"
echo "OUT=${OUT}"
echo "PARTFIELD_CHAMFER_WEIGHT=${PARTFIELD_CHAMFER_WEIGHT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "TARGET_WEIGHT=${TARGET_WEIGHT}"
echo "TARGET_RENDER_WEIGHT=${TARGET_RENDER_WEIGHT}"
echo "N_BUCKETS=${N_BUCKETS}"
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
  --partfield_source_labels "${SOURCE_PARTFIELD_LABELS}" \
  --partfield_target_labels "${TARGET_PARTFIELD_LABELS}" \
  --partfield_labels_aligned \
  --partfield_label_mode auto \
  --partfield_n_buckets "${N_BUCKETS}" \
  --partfield_position_weight "${POSITION_WEIGHT}" \
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
