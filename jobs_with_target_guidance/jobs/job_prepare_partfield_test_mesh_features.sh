#!/bin/bash
#SBATCH --job-name=prep_pf_test
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_test_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/partfield_features/test_meshes

PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_test_meshes}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_test_meshes}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/test_meshes}

if [[ ! -f "${PARTFIELD_REPO}/partfield_inference.py" ]]; then
  echo "Not a PartField checkout: ${PARTFIELD_REPO}"
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
mkdir -p "${PARTFIELD_DATA_DIR}" "${MESHUP_FEATURE_DIR}"

LABELS=(
  "hound_to_bulldog"
  "hound_to_dachshund"
  "hippo_simple"
  "hound_original"
)

MESHES=(
  "${MESHUP_ROOT}/experiments/dog_morphs/outputs/hound_to_bulldog_exp/mesh_final/mesh.obj"
  "${MESHUP_ROOT}/experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj"
  "${MESHUP_ROOT}/experiments/simple_meshes/outputs/hippo_simple/mesh_final/mesh.obj"
  "${MESHUP_ROOT}/meshes/hound.obj"
)

echo "======================================================"
echo "Preparing PartField features for MeshUp test meshes"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "PARTFIELD_DATA_DIR=${PARTFIELD_DATA_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "START_TIME=$(date)"
echo "======================================================"

for idx in "${!MESHES[@]}"; do
  mesh_path="${MESHES[$idx]}"
  label="${LABELS[$idx]}"
  obj_path="${PARTFIELD_DATA_DIR}/${label}.obj"
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

for label in "${LABELS[@]}"; do
  copy_feature "${label}"
done

echo "END_TIME=$(date)"
echo "Prepared PartField features:"
ls -lh "${MESHUP_FEATURE_DIR}"/part_feat_*_0*.npy
