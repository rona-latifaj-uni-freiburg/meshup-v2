#!/bin/bash
#SBATCH --job-name=prep_pf_car
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_car_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_car_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/partfield_features/car_5k

# Point this at the official NVIDIA PartField checkout.
# Example:
#   sbatch --export=ALL,PARTFIELD_REPO=/path/to/PartField,PARTFIELD_ENV=partfield \
#     jobs_with_target_guidance/jobs/job_prepare_partfield_car_features.sh
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_car_5k}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_car_5k}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/car_5k}

if [[ ! -d "${PARTFIELD_REPO}" ]]; then
  echo "PartField repo not found: ${PARTFIELD_REPO}"
  echo "Clone/setup NVIDIA PartField first, or submit with PARTFIELD_REPO=/path/to/PartField."
  exit 1
fi

if [[ ! -f "${PARTFIELD_REPO}/partfield_inference.py" ]]; then
  echo "Not a PartField checkout: ${PARTFIELD_REPO}"
  echo "Expected ${PARTFIELD_REPO}/partfield_inference.py"
  exit 1
fi

if [[ ! -f "${PARTFIELD_CKPT}" ]]; then
  echo "PartField checkpoint not found: ${PARTFIELD_CKPT}"
  echo "Download model_objaverse.ckpt into ${PARTFIELD_REPO}/model/ or submit with PARTFIELD_CKPT=/path/to/checkpoint."
  exit 1
fi

if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
  source "${PARTFIELD_ACTIVATE}"
elif [[ -n "${PARTFIELD_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PARTFIELD_ENV}"
else
  echo "PARTFIELD_ENV/PARTFIELD_ACTIVATE not set; using the current Python environment."
fi

PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
mkdir -p "${PARTFIELD_DATA_DIR}" "${MESHUP_FEATURE_DIR}"

MESHES=(
  "${MESHUP_ROOT}/jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply"
  "${MESHUP_ROOT}/jobs_with_sam3D/meshes/5k_upright_wheels_down/bugatti-centodieci_5k_upright_wheels_down.ply"
  "${MESHUP_ROOT}/jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply"
)

echo "======================================================"
echo "Preparing PartField features for MeshUp car targets"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "PARTFIELD_DATA_DIR=${PARTFIELD_DATA_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "START_TIME=$(date)"
echo "======================================================"

for mesh_path in "${MESHES[@]}"; do
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

copy_feature blueberry_5k_upright_wheels_down
copy_feature bugatti-centodieci_5k_upright_wheels_down
copy_feature santa_fe_5k_upright_wheels_down

echo "END_TIME=$(date)"
echo "Prepared PartField features:"
ls -lh "${MESHUP_FEATURE_DIR}"/part_feat_*_0*.npy
echo
echo "Next, make aligned MeshUp labels/colored segmentations with:"
echo "  sbatch jobs_with_target_guidance/jobs/job_dev_partfield_segment_car_features.sh"
echo "Then run the label-aligned blueberry -> Santa Fe PartField Chamfer deformation with:"
echo "  sbatch jobs_with_target_guidance/jobs/job_dev_blueberry_to_santafe_partfield_chamfer.sh"
