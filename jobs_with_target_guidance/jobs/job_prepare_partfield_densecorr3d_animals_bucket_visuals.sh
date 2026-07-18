#!/bin/bash
#SBATCH --job-name=prep_pf_dense
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_dense_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_dense_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

RUN_NAME=${RUN_NAME:-densecorr3d_animals_20260706}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_densecorr3d_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_densecorr3d_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/densecorr3d_animals/${RUN_NAME}}
SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/densecorr3d_animals/${RUN_NAME}}
VISUAL_DIR=${VISUAL_DIR:-${SEGMENT_OUTPUT_ROOT}/visuals}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-6,7,8,9,10}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
REUSE_FEATURES=${REUSE_FEATURES:-1}

MESH_NAMES=(elephant moose giraffe panther bear cheetah)

mesh_path() {
  case "$1" in
    elephant) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/2d6b3_toy_animals_009/simple_mesh.obj" ;;
    moose) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/1d6d1_toy_animals_015/simple_mesh.obj" ;;
    giraffe) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/34fb4_toy_animals_019/simple_mesh.obj" ;;
    panther) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/071b8_toy_animals_017/simple_mesh.obj" ;;
    bear) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/96615_toy_animals_018/simple_mesh.obj" ;;
    cheetah) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/bdfd0_toy_animals_016/simple_mesh.obj" ;;
    *) echo "Unknown mesh name: $1" >&2; return 1 ;;
  esac
}

feature_path() {
  local name="$1"
  local candidate
  for candidate in \
    "${MESHUP_FEATURE_DIR}/part_feat_${name}_0_batch.npy" \
    "${MESHUP_FEATURE_DIR}/part_feat_${name}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

all_features_exist() {
  local name
  for name in "${MESH_NAMES[@]}"; do
    if ! feature_path "${name}" >/dev/null; then
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

for name in "${MESH_NAMES[@]}"; do
  if [[ ! -f "$(mesh_path "${name}")" ]]; then
    echo "Missing mesh for ${name}: $(mesh_path "${name}")"
    exit 1
  fi
done

mkdir -p "${MESHUP_FEATURE_DIR}" "${SEGMENT_OUTPUT_ROOT}" "${VISUAL_DIR}"
read -r -a BUCKETS <<< "${N_BUCKETS_LIST//,/ }"

echo "======================================================"
echo "Preparing DenseCorr3D animal PartField bucket visuals"
echo "RUN_NAME=${RUN_NAME}"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT}"
echo "VISUAL_DIR=${VISUAL_DIR}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST}"
echo "POSITION_WEIGHT=${POSITION_WEIGHT}"
echo "NORMAL_WEIGHT=${NORMAL_WEIGHT}"
echo "REUSE_FEATURES=${REUSE_FEATURES}"
echo "START_TIME=$(date)"
echo "======================================================"

if [[ "${REUSE_FEATURES}" == "1" ]] && all_features_exist; then
  echo "Reusing existing PartField features in ${MESHUP_FEATURE_DIR}"
else
  if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
    source "${PARTFIELD_ACTIVATE}"
  elif [[ -n "${PARTFIELD_ENV}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${PARTFIELD_ENV}"
  fi

  PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
  mkdir -p "${PARTFIELD_DATA_DIR}"

  for name in "${MESH_NAMES[@]}"; do
    obj_path="${PARTFIELD_DATA_DIR}/${name}.obj"
    echo "Copying $(mesh_path "${name}") -> ${obj_path}"
    cp -f "$(mesh_path "${name}")" "${obj_path}"
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

  for name in "${MESH_NAMES[@]}"; do
    copied=0
    for candidate in \
      "${PARTFIELD_OUTPUT_DIR}/part_feat_${name}_0_batch.npy" \
      "${PARTFIELD_OUTPUT_DIR}/part_feat_${name}_0.npy"; do
      if [[ -f "${candidate}" ]]; then
        cp -f "${candidate}" "${MESHUP_FEATURE_DIR}/"
        echo "Copied ${MESHUP_FEATURE_DIR}/$(basename "${candidate}")"
        copied=1
        break
      fi
    done
    if [[ "${copied}" != "1" ]]; then
      echo "Missing PartField output for ${name} in ${PARTFIELD_OUTPUT_DIR}" >&2
      exit 1
    fi
  done
fi

cd "${MESHUP_ROOT}"

segment_group() {
  local group_slug="$1"
  local group_title="$2"
  shift 2
  local group_names=("$@")
  local group_output_root="${SEGMENT_OUTPUT_ROOT}/${group_slug}"
  local contact_args=()
  local title_args=()
  local name
  local bucket_count

  for name in "${group_names[@]}"; do
    title_args+=("--column-title" "${name}")
  done

  for bucket_count in "${BUCKETS[@]}"; do
    local bucket_token
    bucket_token=$(printf "%02d" "${bucket_count}")
    local scale_output_dir="${group_output_root}/buckets_${bucket_token}"
    local segment_args=()

    for name in "${group_names[@]}"; do
      segment_args+=(
        --mesh "$(mesh_path "${name}")"
        --feature "$(feature_path "${name}")"
        --name "${name}"
      )
    done

    echo
    echo "Writing ${bucket_count}-bucket ${group_slug} labels/colored PLYs to ${scale_output_dir}"
    "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
      "${segment_args[@]}" \
      --output-dir "${scale_output_dir}" \
      --n-buckets "${bucket_count}" \
      --position-weight "${POSITION_WEIGHT}" \
      --normal-weight "${NORMAL_WEIGHT}"

    contact_args+=("--case" "${bucket_count}")
    for name in "${group_names[@]}"; do
      contact_args+=("${scale_output_dir}/colored/${name}_partfield_${bucket_token}_parts.ply")
    done
  done

  local contact_sheet="${VISUAL_DIR}/${group_slug}_partfield_bucket_contact_sheet.png"
  echo "Rendering contact sheet ${contact_sheet}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_multi_bucket_contact_sheet \
    "${contact_args[@]}" \
    "${title_args[@]}" \
    --output "${contact_sheet}" \
    --title "${group_title} PartField bucket comparison" \
    --azimuth -35 \
    --elevation 11 \
    --zoom 0.90 \
    --normalize independent \
    --width 2600 \
    --row-height 360
}

segment_group \
  all_six \
  "DenseCorr3D all six animals" \
  elephant moose giraffe panther bear cheetah

segment_group \
  giraffe_elephant \
  "DenseCorr3D giraffe + elephant" \
  giraffe elephant

segment_group \
  panther_cheetah \
  "DenseCorr3D panther + cheetah" \
  panther cheetah

echo "END_TIME=$(date)"
echo "Feature files:"
find "${MESHUP_FEATURE_DIR}" -maxdepth 1 -type f -name 'part_feat_*' | sort
echo "Visuals:"
find "${VISUAL_DIR}" -maxdepth 1 -type f | sort
