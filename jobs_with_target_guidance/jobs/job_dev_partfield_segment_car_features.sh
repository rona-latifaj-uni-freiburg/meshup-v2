#!/bin/bash
#SBATCH --job-name=seg_pf_car
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=jobs_with_target_guidance/logs/seg_pf_car_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/seg_pf_car_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/partfield_segments

source ./activate_meshup_new.sh

MESH_DIR=${MESH_DIR:-./jobs_with_sam3D/meshes/5k_upright_wheels_down}
PARTFIELD_FEATURE_DIR=${PARTFIELD_FEATURE_DIR:-./jobs_with_target_guidance/partfield_features/car_5k}
OUTPUT_DIR=${OUTPUT_DIR:-./jobs_with_target_guidance/partfield_segments/car_5k}
N_BUCKETS=${N_BUCKETS:-12}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}

feature_path() {
  local base="$1"
  local candidate
  for candidate in \
    "${PARTFIELD_FEATURE_DIR}/part_feat_${base}_0_batch.npy" \
    "${PARTFIELD_FEATURE_DIR}/part_feat_${base}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

BLUEBERRY_MESH="${MESH_DIR}/blueberry_5k_upright_wheels_down.ply"
SANTAFE_MESH="${MESH_DIR}/santa_fe_5k_upright_wheels_down.ply"
BUGATTI_MESH="${MESH_DIR}/bugatti-centodieci_5k_upright_wheels_down.ply"

BLUEBERRY_FEATURE=$(feature_path blueberry_5k_upright_wheels_down || true)
SANTAFE_FEATURE=$(feature_path santa_fe_5k_upright_wheels_down || true)
BUGATTI_FEATURE=$(feature_path bugatti-centodieci_5k_upright_wheels_down || true)

if [[ -z "${BLUEBERRY_FEATURE}" || -z "${SANTAFE_FEATURE}" || -z "${BUGATTI_FEATURE}" ]]; then
  echo "Missing PartField feature files under ${PARTFIELD_FEATURE_DIR}"
  echo "Generate them first with:"
  echo "  sbatch jobs_with_target_guidance/jobs/job_prepare_partfield_car_features.sh"
  exit 1
fi

echo "======================================================"
echo "PartField co-segmentation for MeshUp car meshes"
echo "FEATURE_DIR=${PARTFIELD_FEATURE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "N_BUCKETS=${N_BUCKETS}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST:-${N_BUCKETS}}"
echo "POSITION_WEIGHT=${POSITION_WEIGHT}"
echo "NORMAL_WEIGHT=${NORMAL_WEIGHT}"
echo "START_TIME=$(date)"
echo "======================================================"

if [[ -n "${N_BUCKETS_LIST}" ]]; then
  read -r -a BUCKETS <<< "${N_BUCKETS_LIST//,/ }"
else
  BUCKETS=("${N_BUCKETS}")
fi

for bucket_count in "${BUCKETS[@]}"; do
  if [[ "${bucket_count}" == "${N_BUCKETS}" ]]; then
    SCALE_OUTPUT_DIR="${OUTPUT_DIR}"
  else
    SCALE_OUTPUT_DIR="${OUTPUT_DIR}_${bucket_count}"
  fi

  echo
  echo "Writing ${bucket_count}-bucket labels to ${SCALE_OUTPUT_DIR}"
  python -m jobs_with_target_guidance.partfield_segment \
    --mesh "${BLUEBERRY_MESH}" \
    --feature "${BLUEBERRY_FEATURE}" \
    --name blueberry \
    --mesh "${SANTAFE_MESH}" \
    --feature "${SANTAFE_FEATURE}" \
    --name santa_fe \
    --mesh "${BUGATTI_MESH}" \
    --feature "${BUGATTI_FEATURE}" \
    --name bugatti \
    --output-dir "${SCALE_OUTPUT_DIR}" \
    --n-buckets "${bucket_count}" \
    --position-weight "${POSITION_WEIGHT}" \
    --normal-weight "${NORMAL_WEIGHT}"
done

echo "Use these labels for label-mode Chamfer experiments if desired:"
echo "  ${OUTPUT_DIR}/labels/blueberry_partfield_labels.npz"
echo "  ${OUTPUT_DIR}/labels/santa_fe_partfield_labels.npz"
echo "  ${OUTPUT_DIR}/labels/bugatti_partfield_labels.npz"
if [[ -n "${N_BUCKETS_LIST}" ]]; then
  echo "Multi-scale label dirs:"
  for bucket_count in "${BUCKETS[@]}"; do
    if [[ "${bucket_count}" == "${N_BUCKETS}" ]]; then
      echo "  ${OUTPUT_DIR}"
    else
      echo "  ${OUTPUT_DIR}_${bucket_count}"
    fi
  done
fi
echo "END_TIME=$(date)"
