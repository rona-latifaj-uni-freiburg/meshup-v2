#!/bin/bash
set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"

RUN_NAME=${RUN_NAME:-densecorr3d_animals_20260706}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
FEATURE_DIR=${FEATURE_DIR:-jobs_with_target_guidance/partfield_features/densecorr3d_animals/${RUN_NAME}}
SEGMENT_ROOT=${SEGMENT_ROOT:-jobs_with_target_guidance/partfield_segments/densecorr3d_animals/${RUN_NAME}}
VISUAL_DIR=${VISUAL_DIR:-${SEGMENT_ROOT}/visuals}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}

mesh_path() {
  case "$1" in
    elephant) echo "external/DenseCorr3D/animals/2d6b3_toy_animals_009/simple_mesh.obj" ;;
    moose) echo "external/DenseCorr3D/animals/1d6d1_toy_animals_015/simple_mesh.obj" ;;
    giraffe) echo "external/DenseCorr3D/animals/34fb4_toy_animals_019/simple_mesh.obj" ;;
    panther) echo "external/DenseCorr3D/animals/071b8_toy_animals_017/simple_mesh.obj" ;;
    bear) echo "external/DenseCorr3D/animals/96615_toy_animals_018/simple_mesh.obj" ;;
    cheetah) echo "external/DenseCorr3D/animals/bdfd0_toy_animals_016/simple_mesh.obj" ;;
    *) echo "Unknown mesh name: $1" >&2; return 1 ;;
  esac
}

feature_path() {
  local name="$1"
  local candidate
  for candidate in \
    "${FEATURE_DIR}/part_feat_${name}_0_batch.npy" \
    "${FEATURE_DIR}/part_feat_${name}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Missing feature for ${name} in ${FEATURE_DIR}" >&2
  return 1
}

segment_group_bucket() {
  local group_slug="$1"
  local bucket_count="$2"
  shift 2
  local group_names=("$@")
  local bucket_token
  bucket_token=$(printf "%02d" "${bucket_count}")
  local scale_output_dir="${SEGMENT_ROOT}/${group_slug}/buckets_${bucket_token}"
  local segment_args=()
  local name

  for name in "${group_names[@]}"; do
    segment_args+=(
      --mesh "$(mesh_path "${name}")"
      --feature "$(feature_path "${name}")"
      --name "${name}"
    )
  done

  echo "Writing ${bucket_count}-bucket ${group_slug} labels/colored PLYs to ${scale_output_dir}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
    "${segment_args[@]}" \
    --output-dir "${scale_output_dir}" \
    --n-buckets "${bucket_count}" \
    --position-weight "${POSITION_WEIGHT}" \
    --normal-weight "${NORMAL_WEIGHT}"
}

render_group_sheet() {
  local group_slug="$1"
  local title="$2"
  local buckets_csv="$3"
  shift 3
  local group_names=("$@")
  local contact_args=()
  local title_args=()
  local name
  local bucket_count

  for name in "${group_names[@]}"; do
    title_args+=("--column-title" "${name}")
  done

  read -r -a buckets <<< "${buckets_csv//,/ }"
  for bucket_count in "${buckets[@]}"; do
    local bucket_token
    bucket_token=$(printf "%02d" "${bucket_count}")
    contact_args+=("--case" "${bucket_count}")
    for name in "${group_names[@]}"; do
      contact_args+=("${SEGMENT_ROOT}/${group_slug}/buckets_${bucket_token}/colored/${name}_partfield_${bucket_token}_parts.ply")
    done
  done

  mkdir -p "${VISUAL_DIR}"
  local output="${VISUAL_DIR}/${group_slug}_partfield_bucket_contact_sheet.png"
  echo "Rendering contact sheet ${output}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_multi_bucket_contact_sheet \
    "${contact_args[@]}" \
    "${title_args[@]}" \
    --output "${output}" \
    --title "${title} PartField bucket comparison" \
    --azimuth -35 \
    --elevation 11 \
    --zoom 0.90 \
    --normalize independent \
    --width 2600 \
    --row-height 360
}

segment_group_bucket all_six 5 elephant moose giraffe panther bear cheetah

for bucket_count in 6 7 8; do
  segment_group_bucket cheetah_bear "${bucket_count}" cheetah bear
done

render_group_sheet all_six "DenseCorr3D all six animals" 5,6,7,8,9,10 elephant moose giraffe panther bear cheetah
render_group_sheet cheetah_bear "DenseCorr3D cheetah + bear" 6,7,8 cheetah bear

echo "Visuals:"
find "${VISUAL_DIR}" -maxdepth 1 -type f -name '*.png' | sort
