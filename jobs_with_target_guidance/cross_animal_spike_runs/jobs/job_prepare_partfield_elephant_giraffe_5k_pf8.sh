#!/bin/bash
#SBATCH --job-name=prep_pf_eg5k
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_pf_elephant_giraffe_5k_pf8_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_pf_elephant_giraffe_5k_pf8_%j.err

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"

BASE_DIR=${BASE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/cross_animal_spike_runs}
PREPARED_DIR=${PREPARED_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/densematcher_runs/prepared/requested_animals_20260630}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/densecorr3d_elephant_giraffe_5k_pf8}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/densecorr3d_elephant_giraffe_5k_pf8}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${BASE_DIR}/partfield/features/densecorr3d_elephant_giraffe_5k_pf8}
SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR:-${BASE_DIR}/partfield/segments/densecorr3d_elephant_giraffe_5k_pf8}
N_BUCKETS=${N_BUCKETS:-8}
BUCKET_SUFFIX=$(printf "%02d" "${N_BUCKETS}")
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
BASE_VARIANT=${BASE_VARIANT:-5k}
PF_BASE_VARIANT=${PF_BASE_VARIANT:-pf8}
BALANCED_VARIANT=${BALANCED_VARIANT:-pf8bucketavg}
BALANCE_TARGET_MODE=${BALANCE_TARGET_MODE:-average}
BALANCE_MAX_VERTICES=${BALANCE_MAX_VERTICES:-6000}

SOURCE_OBJECT=2d6b3_toy_animals_009
TARGET_OBJECT=34fb4_toy_animals_019
SOURCE_NAME="${SOURCE_OBJECT}_densecorr3d_${BASE_VARIANT}"
TARGET_NAME="${TARGET_OBJECT}_densecorr3d_${BASE_VARIANT}"
SOURCE_MESH="${PREPARED_DIR}/meshes/${SOURCE_NAME}.obj"
TARGET_MESH="${PREPARED_DIR}/meshes/${TARGET_NAME}.obj"

mkdir -p \
  "${BASE_DIR}/logs" \
  "${MESHUP_FEATURE_DIR}" \
  "${SEGMENT_OUTPUT_DIR}" \
  "${PREPARED_DIR}/meshes" \
  "${PREPARED_DIR}/labels" \
  "${PREPARED_DIR}/colored"

if [[ ! -f "${PARTFIELD_REPO}/partfield_inference.py" ]]; then
  echo "Missing PartField checkout: ${PARTFIELD_REPO}"
  exit 1
fi
if [[ ! -f "${PARTFIELD_CKPT}" ]]; then
  echo "Missing PartField checkpoint: ${PARTFIELD_CKPT}"
  exit 1
fi
for path in "${SOURCE_MESH}" "${TARGET_MESH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing prepared mesh: ${path}"
    exit 1
  fi
done

echo "======================================================"
echo "Preparing PartField-8 elephant/giraffe 5k labels"
echo "PREPARED_DIR=${PREPARED_DIR}"
echo "SOURCE_MESH=${SOURCE_MESH}"
echo "TARGET_MESH=${TARGET_MESH}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR}"
echo "PF_BASE_VARIANT=${PF_BASE_VARIANT}"
echo "BALANCED_VARIANT=${BALANCED_VARIANT}"
echo "START_TIME=$(date)"
echo "======================================================"

if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
  source "${PARTFIELD_ACTIVATE}"
elif [[ -n "${PARTFIELD_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PARTFIELD_ENV}"
fi

PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
mkdir -p "${PARTFIELD_DATA_DIR}"
find "${PARTFIELD_DATA_DIR}" -maxdepth 1 -type f \( -name '*.obj' -o -name '*.mtl' -o -name '*.obj.mtl' \) -delete

for entry in "${SOURCE_NAME}:${SOURCE_MESH}" "${TARGET_NAME}:${TARGET_MESH}"; do
  label="${entry%%:*}"
  mesh_path="${entry#*:}"
  obj_path="${PARTFIELD_DATA_DIR}/${label}.obj"
  echo "Converting ${mesh_path} -> ${obj_path}"
  python -c "import pymeshlab, sys; ms=pymeshlab.MeshSet(); ms.load_new_mesh(sys.argv[1]); ms.save_current_mesh(sys.argv[2], save_vertex_color=False)" "${mesh_path}" "${obj_path}"
  python -c "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(''.join(line for line in p.read_text().splitlines(True) if not line.startswith(('mtllib ', 'usemtl '))))" "${obj_path}"
  find "${PARTFIELD_DATA_DIR}" -maxdepth 1 -type f \( -name "${label}.mtl" -o -name "${label}.obj.mtl" \) -delete
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

copy_feature "${SOURCE_NAME}"
copy_feature "${TARGET_NAME}"

cd "${MESHUP_ROOT}"
source ./activate_meshup_new.sh

feature_path() {
  local base="$1"
  local candidate
  for candidate in \
    "${MESHUP_FEATURE_DIR}/part_feat_${base}_0_batch.npy" \
    "${MESHUP_FEATURE_DIR}/part_feat_${base}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

python -m jobs_with_target_guidance.partfield_segment \
  --mesh "${SOURCE_MESH}" \
  --feature "$(feature_path "${SOURCE_NAME}")" \
  --name "${SOURCE_NAME}" \
  --mesh "${TARGET_MESH}" \
  --feature "$(feature_path "${TARGET_NAME}")" \
  --name "${TARGET_NAME}" \
  --output-dir "${SEGMENT_OUTPUT_DIR}" \
  --n-buckets "${N_BUCKETS}" \
  --position-weight "${POSITION_WEIGHT}" \
  --normal-weight "${NORMAL_WEIGHT}"

cp -f "${SOURCE_MESH}" "${PREPARED_DIR}/meshes/${SOURCE_OBJECT}_densecorr3d_${PF_BASE_VARIANT}.obj"
cp -f "${TARGET_MESH}" "${PREPARED_DIR}/meshes/${TARGET_OBJECT}_densecorr3d_${PF_BASE_VARIANT}.obj"
cp -f "${SEGMENT_OUTPUT_DIR}/labels/${SOURCE_NAME}_partfield_labels.npz" "${PREPARED_DIR}/labels/${SOURCE_OBJECT}_densecorr3d_${PF_BASE_VARIANT}_labels.npz"
cp -f "${SEGMENT_OUTPUT_DIR}/labels/${TARGET_NAME}_partfield_labels.npz" "${PREPARED_DIR}/labels/${TARGET_OBJECT}_densecorr3d_${PF_BASE_VARIANT}_labels.npz"
cp -f "${SEGMENT_OUTPUT_DIR}/colored/${SOURCE_NAME}_partfield_${BUCKET_SUFFIX}_parts.ply" "${PREPARED_DIR}/colored/${SOURCE_OBJECT}_densecorr3d_${PF_BASE_VARIANT}_groups.ply"
cp -f "${SEGMENT_OUTPUT_DIR}/colored/${TARGET_NAME}_partfield_${BUCKET_SUFFIX}_parts.ply" "${PREPARED_DIR}/colored/${TARGET_OBJECT}_densecorr3d_${PF_BASE_VARIANT}_groups.ply"

python jobs_with_target_guidance/densecorr3d_balance_bucket_counts.py \
  --prepared-dir "${PREPARED_DIR}" \
  --source-object "${SOURCE_OBJECT}" \
  --target-object "${TARGET_OBJECT}" \
  --base-variant "${PF_BASE_VARIANT}" \
  --target-mode "${BALANCE_TARGET_MODE}" \
  --max-vertices "${BALANCE_MAX_VERTICES}" \
  --output-variant "${BALANCED_VARIANT}" \
  --overwrite

echo "END_TIME=$(date)"
echo "PartField features: ${MESHUP_FEATURE_DIR}"
echo "PartField-8 labels: ${SEGMENT_OUTPUT_DIR}"
echo "Prepared balanced variant: ${BALANCED_VARIANT}"
