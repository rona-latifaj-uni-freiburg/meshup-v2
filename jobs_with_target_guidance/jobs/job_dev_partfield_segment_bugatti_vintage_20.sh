#!/bin/bash
#SBATCH --job-name=seg_pf_bug_v20
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=jobs_with_target_guidance/logs/seg_pf_bug_v20_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/seg_pf_bug_v20_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/partfield_segments

source ./activate_meshup_new.sh

MESH_DIR=${MESH_DIR:-./jobs_with_sam3D/meshes/5k_upright_wheels_down}
PARTFIELD_FEATURE_DIR=${PARTFIELD_FEATURE_DIR:-./jobs_with_target_guidance/partfield_features/bugatti_vintage_5k}
OUTPUT_DIR=${OUTPUT_DIR:-./jobs_with_target_guidance/partfield_segments/bugatti_vintage_5k_20}
N_BUCKETS=${N_BUCKETS:-20}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}

BUGATTI_MESH="${MESH_DIR}/bugatti-centodieci_5k_upright_wheels_down.ply"
VINTAGE_MESH="${MESH_DIR}/vintage_car_5k_upright_wheels_down.ply"
BUGATTI_FEATURE="${PARTFIELD_FEATURE_DIR}/part_feat_bugatti-centodieci_5k_upright_wheels_down_0_batch.npy"
VINTAGE_FEATURE="${PARTFIELD_FEATURE_DIR}/part_feat_vintage_car_5k_upright_wheels_down_0_batch.npy"

if [[ ! -f "${BUGATTI_FEATURE}" || ! -f "${VINTAGE_FEATURE}" ]]; then
  echo "Missing Bugatti/Vintage PartField feature files."
  echo "Expected: ${BUGATTI_FEATURE}"
  echo "Expected: ${VINTAGE_FEATURE}"
  echo "Generate them first with:"
  echo "  sbatch jobs_with_target_guidance/jobs/job_prepare_partfield_bugatti_vintage_features.sh"
  exit 1
fi

echo "======================================================"
echo "20-bucket PartField co-segmentation for Bugatti/Vintage"
echo "FEATURE_DIR=${PARTFIELD_FEATURE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "N_BUCKETS=${N_BUCKETS}"
echo "POSITION_WEIGHT=${POSITION_WEIGHT}"
echo "NORMAL_WEIGHT=${NORMAL_WEIGHT}"
echo "START_TIME=$(date)"
echo "======================================================"

python -m jobs_with_target_guidance.partfield_segment \
  --mesh "${BUGATTI_MESH}" \
  --feature "${BUGATTI_FEATURE}" \
  --name bugatti \
  --mesh "${VINTAGE_MESH}" \
  --feature "${VINTAGE_FEATURE}" \
  --name vintage_car \
  --output-dir "${OUTPUT_DIR}" \
  --n-buckets "${N_BUCKETS}" \
  --position-weight "${POSITION_WEIGHT}" \
  --normal-weight "${NORMAL_WEIGHT}"

echo "END_TIME=$(date)"
echo "Use these labels for 20-bucket ablations:"
echo "  ${OUTPUT_DIR}/labels/bugatti_partfield_labels.npz"
echo "  ${OUTPUT_DIR}/labels/vintage_car_partfield_labels.npz"
