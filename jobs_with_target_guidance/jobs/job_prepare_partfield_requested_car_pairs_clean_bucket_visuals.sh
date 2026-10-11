#!/bin/bash
#SBATCH --job-name=prep_pf_carpairs
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_carpairs_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_carpairs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

RUN_NAME=${RUN_NAME:-requested_car_pairs_clean_20260722}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/${RUN_NAME}}
PAIR_SEGMENT_ROOT=${PAIR_SEGMENT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/${RUN_NAME}}
PAIR_VISUAL_DIR=${PAIR_VISUAL_DIR:-${PAIR_SEGMENT_ROOT}/visuals}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-6,7,8,9,10}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
REUSE_FEATURES=${REUSE_FEATURES:-1}

BASE_CLEAN_DIR="${MESHUP_ROOT}/jobs_with_target_guidance/component_analysis/car_5k/largest_components"
NEW_CLEAN_DIR="${MESHUP_ROOT}/jobs_with_target_guidance/component_analysis/new_car_5k/largest_components"

MESH_KEYS=(
  f1_car
  f1_verstappen
  g_class
  green_suv
  pink_samrt
  red_smart
  red_truck1
  red_truck3
  mini_cooper
  kona
  blueberry
  santa_fe
)

mesh_path() {
  local key="$1"
  case "${key}" in
    f1_car) echo "${NEW_CLEAN_DIR}/f1_car_5k_upright_wheels_down_largest_component.ply" ;;
    f1_verstappen) echo "${NEW_CLEAN_DIR}/f1_verstappen_5k_upright_wheels_down_largest_component.ply" ;;
    g_class) echo "${NEW_CLEAN_DIR}/g_class_5k_upright_wheels_down_largest_component.ply" ;;
    green_suv) echo "${NEW_CLEAN_DIR}/green_suv_5k_upright_wheels_down_largest_component.ply" ;;
    pink_samrt) echo "${NEW_CLEAN_DIR}/pink_samrt_5k_upright_wheels_down_largest_component.ply" ;;
    red_smart) echo "${NEW_CLEAN_DIR}/red_smart_5k_upright_wheels_down_largest_component.ply" ;;
    red_truck1) echo "${NEW_CLEAN_DIR}/red_truck1_5k_upright_wheels_down_largest_component.ply" ;;
    red_truck3) echo "${NEW_CLEAN_DIR}/red_truck3_5k_upright_wheels_down_largest_component.ply" ;;
    mini_cooper) echo "${NEW_CLEAN_DIR}/mini_cooper_5k_upright_wheels_down_largest_component.ply" ;;
    kona) echo "${BASE_CLEAN_DIR}/kona_5k_upright_wheels_down_largest_component.ply" ;;
    blueberry) echo "${BASE_CLEAN_DIR}/blueberry_5k_upright_wheels_down_largest_component.ply" ;;
    santa_fe) echo "${BASE_CLEAN_DIR}/santa_fe_5k_upright_wheels_down_largest_component.ply" ;;
    *) echo "Unknown mesh key: ${key}" >&2; return 1 ;;
  esac
}

label_name() {
  local key="$1"
  case "${key}" in
    pink_samrt) echo "pink_smart" ;;
    *) echo "${key}" ;;
  esac
}

feature_base() {
  basename "$(mesh_path "$1")" .ply
}

feature_path() {
  local key="$1"
  local base
  base="$(feature_base "${key}")"
  local candidate
  for candidate in \
    "${MESHUP_FEATURE_DIR}/part_feat_${base}_0_batch.npy" \
    "${MESHUP_FEATURE_DIR}/part_feat_${base}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Missing feature for ${key} (${base}) in ${MESHUP_FEATURE_DIR}" >&2
  return 1
}

all_features_exist() {
  local key
  for key in "${MESH_KEYS[@]}"; do
    if ! feature_path "${key}" >/dev/null; then
      return 1
    fi
  done
  return 0
}

if [[ ! -d "${PARTFIELD_REPO}" || ! -f "${PARTFIELD_REPO}/partfield_inference.py" ]]; then
  echo "PartField checkout not found: ${PARTFIELD_REPO}"
  exit 1
fi

if [[ ! -f "${PARTFIELD_CKPT}" ]]; then
  echo "PartField checkpoint not found: ${PARTFIELD_CKPT}"
  exit 1
fi

if [[ ! -x "${MESHUP_PY}" ]]; then
  echo "MeshUp Python not executable: ${MESHUP_PY}"
  exit 1
fi

for key in "${MESH_KEYS[@]}"; do
  if [[ ! -f "$(mesh_path "${key}")" ]]; then
    echo "Missing cleaned mesh for ${key}: $(mesh_path "${key}")"
    exit 1
  fi
done

mkdir -p "${MESHUP_FEATURE_DIR}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}"

echo "======================================================"
echo "Preparing clean largest-component PartField visuals for requested car pairs"
echo "RUN_NAME=${RUN_NAME}"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "PAIR_SEGMENT_ROOT=${PAIR_SEGMENT_ROOT}"
echo "PAIR_VISUAL_DIR=${PAIR_VISUAL_DIR}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST}"
echo "START_TIME=$(date)"
echo "======================================================"

if [[ "${REUSE_FEATURES}" == "1" ]] && all_features_exist; then
  echo "Reusing existing clean-car PartField features in ${MESHUP_FEATURE_DIR}"
else
  if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
    source "${PARTFIELD_ACTIVATE}"
  elif [[ -n "${PARTFIELD_ENV}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${PARTFIELD_ENV}"
  fi

  PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
  mkdir -p "${PARTFIELD_DATA_DIR}"

  for key in "${MESH_KEYS[@]}"; do
    mesh_file="$(mesh_path "${key}")"
    base="$(feature_base "${key}")"
    obj_path="${PARTFIELD_DATA_DIR}/${base}.obj"
    echo "Converting ${mesh_file} -> ${obj_path}"
    python -c "import pymeshlab, sys; ms=pymeshlab.MeshSet(); ms.load_new_mesh(sys.argv[1]); ms.save_current_mesh(sys.argv[2], save_vertex_color=False)" "${mesh_file}" "${obj_path}"
  done

  cd "${PARTFIELD_REPO}"
  python partfield_inference.py \
    -c configs/final/demo.yaml \
    --opts \
    continue_ckpt "${PARTFIELD_CKPT}" \
    result_name "${PARTFIELD_RESULT_NAME}" \
    dataset.data_path "${PARTFIELD_DATA_SUBDIR}" \
    n_point_per_face "${PARTFIELD_N_POINT_PER_FACE}" \
    n_sample_each "${PARTFIELD_N_SAMPLE_EACH}" \
    dataset.val_num_workers "${PARTFIELD_VAL_NUM_WORKERS}"

  PARTFIELD_OUTPUT_DIR="${PARTFIELD_REPO}/exp_results/${PARTFIELD_RESULT_NAME}"
  for key in "${MESH_KEYS[@]}"; do
    base="$(feature_base "${key}")"
    copied=0
    for candidate in \
      "${PARTFIELD_OUTPUT_DIR}/part_feat_${base}_0_batch.npy" \
      "${PARTFIELD_OUTPUT_DIR}/part_feat_${base}_0.npy"; do
      if [[ -f "${candidate}" ]]; then
        cp -f "${candidate}" "${MESHUP_FEATURE_DIR}/"
        echo "Copied ${MESHUP_FEATURE_DIR}/$(basename "${candidate}")"
        copied=1
        break
      fi
    done
    if [[ "${copied}" != "1" ]]; then
      echo "Missing PartField output for ${base} in ${PARTFIELD_OUTPUT_DIR}" >&2
      exit 1
    fi
  done

  cd "${MESHUP_ROOT}"
fi

segment_pair_series() {
  local pair_slug="$1"
  local left_key="$2"
  local right_key="$3"
  local title="$4"

  local left_label
  local right_label
  left_label="$(label_name "${left_key}")"
  right_label="$(label_name "${right_key}")"

  read -r -a buckets <<< "${N_BUCKETS_LIST//,/ }"
  local contact_args=()
  local bucket_count
  for bucket_count in "${buckets[@]}"; do
    local bucket_token
    bucket_token=$(printf "%02d" "${bucket_count}")
    local scale_output_dir="${PAIR_SEGMENT_ROOT}/${pair_slug}/buckets_${bucket_token}"

    echo
    echo "Writing ${bucket_count}-bucket ${pair_slug} labels/colored PLYs to ${scale_output_dir}"
    "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
      --mesh "$(mesh_path "${left_key}")" \
      --feature "$(feature_path "${left_key}")" \
      --name "${left_label}" \
      --mesh "$(mesh_path "${right_key}")" \
      --feature "$(feature_path "${right_key}")" \
      --name "${right_label}" \
      --output-dir "${scale_output_dir}" \
      --n-buckets "${bucket_count}" \
      --position-weight "${POSITION_WEIGHT}" \
      --normal-weight "${NORMAL_WEIGHT}"

    contact_args+=(
      "--case" "${bucket_count}"
      "${scale_output_dir}/colored/${left_label}_partfield_${bucket_token}_parts.ply"
      "${scale_output_dir}/colored/${right_label}_partfield_${bucket_token}_parts.ply"
    )
  done

  local contact_sheet="${PAIR_VISUAL_DIR}/${pair_slug}_partfield_bucket_contact_sheet.png"
  echo "Rendering contact sheet ${contact_sheet}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_bucket_contact_sheet \
    "${contact_args[@]}" \
    --output "${contact_sheet}" \
    --title "${title}" \
    --left-title "${left_label}" \
    --right-title "${right_label}" \
    --azimuth -35 \
    --elevation 11 \
    --zoom 0.90 \
    --normalize independent
}

segment_pair_series f1_car_to_f1_verstappen f1_car f1_verstappen "f1_car -> f1_verstappen clean PartField bucket comparison"
segment_pair_series g_class_to_green_suv g_class green_suv "g_class -> green_suv clean PartField bucket comparison"
segment_pair_series pink_smart_to_red_smart pink_samrt red_smart "pink_smart -> red_smart clean PartField bucket comparison"
segment_pair_series red_truck1_to_red_truck3 red_truck1 red_truck3 "red_truck1 -> red_truck3 clean PartField bucket comparison"
segment_pair_series mini_cooper_to_pink_smart mini_cooper pink_samrt "mini_cooper -> pink_smart clean PartField bucket comparison"
segment_pair_series kona_to_g_class kona g_class "kona -> g_class clean PartField bucket comparison"
segment_pair_series blueberry_to_santa_fe blueberry santa_fe "blueberry -> santa_fe clean PartField bucket comparison"

echo "END_TIME=$(date)"
echo "Pair visuals:"
find "${PAIR_VISUAL_DIR}" -maxdepth 1 -type f | sort
