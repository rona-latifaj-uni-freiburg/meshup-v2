#!/bin/bash
#SBATCH --job-name=dev_blue_sfe_pf2k
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/dev_blue_sfe_pf2k_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/dev_blue_sfe_pf2k_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

SOURCE=./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply
TARGET=./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply
LABEL=blueberry_to_santafe_suv_2k_dev
PROMPT="an SUV"

PARTFIELD_SEGMENT_DIR=${PARTFIELD_SEGMENT_DIR:-./jobs_with_target_guidance/partfield_segments/car_5k}
PARTFIELD_LABEL_DIR=${PARTFIELD_LABEL_DIR:-${PARTFIELD_SEGMENT_DIR}/labels}
SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS:-${PARTFIELD_LABEL_DIR}/blueberry_partfield_labels.npz}
TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS:-${PARTFIELD_LABEL_DIR}/santa_fe_partfield_labels.npz}
SOURCE_PARTFIELD_COLORED=${SOURCE_PARTFIELD_COLORED:-${PARTFIELD_SEGMENT_DIR}/colored/blueberry_partfield_12_parts.ply}
TARGET_PARTFIELD_COLORED=${TARGET_PARTFIELD_COLORED:-${PARTFIELD_SEGMENT_DIR}/colored/santa_fe_partfield_12_parts.ply}

if [[ ! -f "${SOURCE_PARTFIELD_LABELS}" || ! -f "${TARGET_PARTFIELD_LABELS}" ]]; then
  echo "Missing aligned PartField label files."
  echo "Expected source labels: ${SOURCE_PARTFIELD_LABELS}"
  echo "Expected target labels: ${TARGET_PARTFIELD_LABELS}"
  echo "Generate them with:"
  echo "  sbatch jobs_with_target_guidance/jobs/job_dev_partfield_segment_car_features.sh"
  exit 1
fi

PARTFIELD_CHAMFER_WEIGHT=7000
GLOBAL_CHAMFER_WEIGHT=750
TARGET_WEIGHT=130
TARGET_RENDER_WEIGHT=8
REG_WEIGHT=1800
N_BUCKETS=12
POSITION_WEIGHT=0.05
EPOCHS=2000

OUT=./jobs_with_target_guidance/outputs/${LABEL}_partfield_chamfer_${SLURM_JOB_ID:-manual}

echo "======================================================"
echo "DEV H100 blueberry -> Santa Fe PartField target Chamfer, ${EPOCHS} epochs"
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
