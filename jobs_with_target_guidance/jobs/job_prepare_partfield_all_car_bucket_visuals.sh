#!/bin/bash
#SBATCH --job-name=prep_pf_allcars_vis
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_allcars_vis_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_allcars_vis_%j.err
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
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_all_car_bucket_visuals_5k_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_all_car_bucket_visuals_5k_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/all_car_bucket_visuals_5k}
SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/all_car_bucket_visuals_5k}
N_BUCKETS=${N_BUCKETS:-12}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-8,12,20}
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

BASE_CAR_DIR="${MESHUP_ROOT}/jobs_with_sam3D/meshes/5k_upright_wheels_down"
NEW_CAR_DIR="${MESHUP_ROOT}/jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down"

MESHES=(
  "${BASE_CAR_DIR}/blueberry_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/bugatti-centodieci_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/kona_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/old_car_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/oldie_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/passati_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/red_car_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/santa_fe_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/usa_suv_5k_upright_wheels_down.ply"
  "${BASE_CAR_DIR}/vintage_car_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/cars_doc_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/f1_car_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/f1_verstappen_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/g_class_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/green_suv_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/mini_cooper_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/no_roof_car_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/pink_samrt_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/red_smart_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/red_truck1_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/red_truck2_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/red_truck3_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/red_truck4_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/venom_qt_5k_upright_wheels_down.ply"
  "${NEW_CAR_DIR}/white_longer_mini_cargo_truck_5k_upright_wheels_down.ply"
)

NAMES=(
  blueberry
  bugatti_centodieci
  kona
  old_car
  oldie
  passati
  red_car
  santa_fe
  usa_suv
  vintage_car
  cars_doc
  f1_car
  f1_verstappen
  g_class
  green_suv
  mini_cooper
  no_roof_car
  pink_samrt
  red_smart
  red_truck1
  red_truck2
  red_truck3
  red_truck4
  venom_qt
  white_longer_mini_cargo_truck
)

if [[ "${#MESHES[@]}" -ne "${#NAMES[@]}" ]]; then
  echo "Internal error: MESHES/NAMES length mismatch."
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

echo "======================================================"
echo "Preparing 8/12/20 PartField bucket visuals for all car meshes"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "PARTFIELD_DATA_DIR=${PARTFIELD_DATA_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST}"
echo "N_MESHES=${#MESHES[@]}"
echo "START_TIME=$(date)"
echo "======================================================"

for mesh_path in "${MESHES[@]}"; do
  if [[ ! -f "${mesh_path}" ]]; then
    echo "Missing mesh: ${mesh_path}"
    exit 1
  fi
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
      echo "${MESHUP_FEATURE_DIR}/$(basename "${candidate}")"
      return 0
    fi
  done
  echo "Missing PartField output for ${base} in ${PARTFIELD_OUTPUT_DIR}" >&2
  return 1
}

FEATURES=()
for mesh_path in "${MESHES[@]}"; do
  base=$(basename "${mesh_path%.*}")
  FEATURES+=("$(copy_feature "${base}")")
done

cd "${MESHUP_ROOT}"
source ./activate_meshup_new.sh

SEG_ARGS=()
for idx in "${!MESHES[@]}"; do
  SEG_ARGS+=("--mesh" "${MESHES[$idx]}")
  SEG_ARGS+=("--feature" "${FEATURES[$idx]}")
  SEG_ARGS+=("--name" "${NAMES[$idx]}")
done

read -r -a BUCKETS <<< "${N_BUCKETS_LIST//,/ }"

for bucket_count in "${BUCKETS[@]}"; do
  if [[ "${bucket_count}" == "${N_BUCKETS}" ]]; then
    SCALE_OUTPUT_DIR="${SEGMENT_OUTPUT_DIR}"
  else
    SCALE_OUTPUT_DIR="${SEGMENT_OUTPUT_DIR}_${bucket_count}"
  fi

  echo
  echo "Writing ${bucket_count}-bucket labels/colored PLYs to ${SCALE_OUTPUT_DIR}"
  python -m jobs_with_target_guidance.partfield_segment \
    "${SEG_ARGS[@]}" \
    --output-dir "${SCALE_OUTPUT_DIR}" \
    --n-buckets "${bucket_count}" \
    --position-weight "${POSITION_WEIGHT}" \
    --normal-weight "${NORMAL_WEIGHT}"
done

echo "END_TIME=$(date)"
echo "Colored PartField bucket PLYs:"
echo "  ${SEGMENT_OUTPUT_DIR}_8/colored"
echo "  ${SEGMENT_OUTPUT_DIR}/colored"
echo "  ${SEGMENT_OUTPUT_DIR}_20/colored"
