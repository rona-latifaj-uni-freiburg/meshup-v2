#!/bin/bash
#SBATCH --job-name=prep_pf_bug_vint
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_bug_vint_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_bug_vint_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_bugatti_vintage_5k}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_bugatti_vintage_5k}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/bugatti_vintage_5k}
SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/bugatti_vintage_5k}
N_BUCKETS=${N_BUCKETS:-12}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}

if [[ ! -d "${PARTFIELD_REPO}" || ! -f "${PARTFIELD_REPO}/partfield_inference.py" ]]; then
  echo "PartField checkout not found: ${PARTFIELD_REPO}"
  exit 1
fi

if [[ ! -f "${PARTFIELD_CKPT}" ]]; then
  echo "PartField checkpoint not found: ${PARTFIELD_CKPT}"
  exit 1
fi

if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
  source "${PARTFIELD_ACTIVATE}"
elif [[ -n "${PARTFIELD_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PARTFIELD_ENV}"
fi

PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
mkdir -p "${PARTFIELD_DATA_DIR}" "${MESHUP_FEATURE_DIR}" "${SEGMENT_OUTPUT_DIR}"

BUGATTI_MESH="${MESHUP_ROOT}/jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply"
VINTAGE_MESH="${MESHUP_ROOT}/jobs_with_sam3D/meshes/5k_upright_wheels_down/vintage_car_5k_upright_wheels_down.ply"

echo "======================================================"
echo "Preparing and co-segmenting PartField features for Bugatti/Vintage"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "PARTFIELD_DATA_DIR=${PARTFIELD_DATA_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR}"
echo "N_BUCKETS=${N_BUCKETS}"
echo "START_TIME=$(date)"
echo "======================================================"

for mesh_path in "${BUGATTI_MESH}" "${VINTAGE_MESH}"; do
  base=$(basename "${mesh_path%.*}")
  obj_path="${PARTFIELD_DATA_DIR}/${base}.obj"
  echo "Converting ${mesh_path} -> ${obj_path}"
  python -c "import pymeshlab, sys; ms=pymeshlab.MeshSet(); ms.load_new_mesh(sys.argv[1]); ms.save_current_mesh(sys.argv[2], save_vertex_color=False)" "${mesh_path}" "${obj_path}"
done

cd "${PARTFIELD_REPO}"

python partfield_inference.py \
  -c configs/final/demo.yaml \
  --opts \
  continue_ckpt "${PARTFIELD_CKPT}" \
  result_name "${PARTFIELD_RESULT_NAME}" \
  dataset.data_path "${PARTFIELD_DATA_SUBDIR}" \
  n_point_per_face "${PARTFIELD_N_POINT_PER_FACE}" \
  n_sample_each "${PARTFIELD_N_SAMPLE_EACH}"

PARTFIELD_OUTPUT_DIR="${PARTFIELD_REPO}/exp_results/${PARTFIELD_RESULT_NAME}"

copy_feature() {
  local base="$1"
  local candidate
  for candidate in \
    "${PARTFIELD_OUTPUT_DIR}/part_feat_${base}_0_batch.npy" \
    "${PARTFIELD_OUTPUT_DIR}/part_feat_${base}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      cp -f "${candidate}" "${MESHUP_FEATURE_DIR}/"
      echo "Copied $(basename "${candidate}")"
      return 0
    fi
  done
  echo "Missing PartField output for ${base} in ${PARTFIELD_OUTPUT_DIR}"
  return 1
}

copy_feature bugatti-centodieci_5k_upright_wheels_down
copy_feature vintage_car_5k_upright_wheels_down

cd "${MESHUP_ROOT}"
source ./activate_meshup_new.sh

python -m jobs_with_target_guidance.partfield_segment \
  --mesh "${BUGATTI_MESH}" \
  --feature "${MESHUP_FEATURE_DIR}/part_feat_bugatti-centodieci_5k_upright_wheels_down_0_batch.npy" \
  --name bugatti \
  --mesh "${VINTAGE_MESH}" \
  --feature "${MESHUP_FEATURE_DIR}/part_feat_vintage_car_5k_upright_wheels_down_0_batch.npy" \
  --name vintage_car \
  --output-dir "${SEGMENT_OUTPUT_DIR}" \
  --n-buckets "${N_BUCKETS}" \
  --position-weight "${POSITION_WEIGHT}" \
  --normal-weight "${NORMAL_WEIGHT}"

echo "END_TIME=$(date)"
echo "Prepared aligned PartField labels:"
echo "  ${SEGMENT_OUTPUT_DIR}/labels/bugatti_partfield_labels.npz"
echo "  ${SEGMENT_OUTPUT_DIR}/labels/vintage_car_partfield_labels.npz"
echo "Colored segmentations:"
echo "  ${SEGMENT_OUTPUT_DIR}/colored/bugatti_partfield_12_parts.ply"
echo "  ${SEGMENT_OUTPUT_DIR}/colored/vintage_car_partfield_12_parts.ply"
