#!/bin/bash
#SBATCH --job-name=prep_pf_animals
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_animals_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_animals_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

RUN_NAME=${RUN_NAME:-requested_pairs_20260706}
MESH_DIR=${MESH_DIR:-${MESHUP_ROOT}/mesh_creator_for_meshup/sam3D/processed_meshes}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_sam3d_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_sam3d_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/sam3d_animals/${RUN_NAME}}
SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/sam3d_animals/${RUN_NAME}}
VISUAL_DIR=${VISUAL_DIR:-${SEGMENT_OUTPUT_ROOT}/visuals}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-7,8,9,10}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
RENDER_VIDEOS=${RENDER_VIDEOS:-0}

MESH_BASES=(
  cat2
  Chihuahua2
  horse2
  giraffe4
  panda3
  bear
  panda2
  fox
)

PAIR_SLUGS=(
  cat2_to_chihuahua2
  horse2_to_giraffe4
  panda3_to_bear
  panda2_to_panda3
  fox_to_cat2
)

PAIR_LEFT_BASES=(
  cat2
  horse2
  panda3
  panda2
  fox
)

PAIR_RIGHT_BASES=(
  Chihuahua2
  giraffe4
  bear
  panda3
  cat2
)

PAIR_LEFT_NAMES=(
  cat2
  horse2
  panda3
  panda2
  fox
)

PAIR_RIGHT_NAMES=(
  chihuahua2
  giraffe4
  bear
  panda3
  cat2
)

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

for base in "${MESH_BASES[@]}"; do
  if [[ ! -f "${MESH_DIR}/${base}.ply" ]]; then
    echo "Missing mesh: ${MESH_DIR}/${base}.ply"
    exit 1
  fi
done

if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
  source "${PARTFIELD_ACTIVATE}"
elif [[ -n "${PARTFIELD_ENV}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PARTFIELD_ENV}"
fi

PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
mkdir -p "${PARTFIELD_DATA_DIR}" "${MESHUP_FEATURE_DIR}" "${SEGMENT_OUTPUT_ROOT}" "${VISUAL_DIR}"

echo "======================================================"
echo "Preparing PartField 7/8/9/10 bucket visuals for requested SAM3D animal pairs"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "PARTFIELD_DATA_DIR=${PARTFIELD_DATA_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT}"
echo "VISUAL_DIR=${VISUAL_DIR}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST}"
echo "POSITION_WEIGHT=${POSITION_WEIGHT}"
echo "NORMAL_WEIGHT=${NORMAL_WEIGHT}"
echo "START_TIME=$(date)"
echo "======================================================"

for base in "${MESH_BASES[@]}"; do
  mesh_path="${MESH_DIR}/${base}.ply"
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
  n_sample_each "${PARTFIELD_N_SAMPLE_EACH}" \
  dataset.val_num_workers "${PARTFIELD_VAL_NUM_WORKERS}"

PARTFIELD_OUTPUT_DIR="${PARTFIELD_REPO}/exp_results/${PARTFIELD_RESULT_NAME}"

copy_feature() {
  local base="$1"
  local candidate
  for candidate in \
    "${PARTFIELD_OUTPUT_DIR}/part_feat_${base}_0_batch.npy" \
    "${PARTFIELD_OUTPUT_DIR}/part_feat_${base}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      cp -f "${candidate}" "${MESHUP_FEATURE_DIR}/"
      echo "Copied ${MESHUP_FEATURE_DIR}/$(basename "${candidate}")"
      return 0
    fi
  done
  echo "Missing PartField output for ${base} in ${PARTFIELD_OUTPUT_DIR}" >&2
  return 1
}

for base in "${MESH_BASES[@]}"; do
  copy_feature "${base}"
done

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

cd "${MESHUP_ROOT}"
read -r -a BUCKETS <<< "${N_BUCKETS_LIST//,/ }"

for pair_idx in "${!PAIR_SLUGS[@]}"; do
  pair_slug="${PAIR_SLUGS[$pair_idx]}"
  left_base="${PAIR_LEFT_BASES[$pair_idx]}"
  right_base="${PAIR_RIGHT_BASES[$pair_idx]}"
  left_name="${PAIR_LEFT_NAMES[$pair_idx]}"
  right_name="${PAIR_RIGHT_NAMES[$pair_idx]}"
  pair_output_root="${SEGMENT_OUTPUT_ROOT}/${pair_slug}"
  contact_args=()

  for bucket_count in "${BUCKETS[@]}"; do
    bucket_token=$(printf "%02d" "${bucket_count}")
    scale_output_dir="${pair_output_root}/buckets_${bucket_token}"

    echo
    echo "Writing ${bucket_count}-bucket labels/colored PLYs to ${scale_output_dir}"
    "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
      --mesh "${MESH_DIR}/${left_base}.ply" \
      --feature "$(feature_path "${left_base}")" \
      --name "${left_name}" \
      --mesh "${MESH_DIR}/${right_base}.ply" \
      --feature "$(feature_path "${right_base}")" \
      --name "${right_name}" \
      --output-dir "${scale_output_dir}" \
      --n-buckets "${bucket_count}" \
      --position-weight "${POSITION_WEIGHT}" \
      --normal-weight "${NORMAL_WEIGHT}"

    left_colored="${scale_output_dir}/colored/${left_name}_partfield_${bucket_token}_parts.ply"
    right_colored="${scale_output_dir}/colored/${right_name}_partfield_${bucket_token}_parts.ply"
    contact_args+=("--case" "${bucket_count}" "${left_colored}" "${right_colored}")

    if [[ "${RENDER_VIDEOS}" == "1" ]]; then
      video_out="${VISUAL_DIR}/${pair_slug}_partfield_buckets_${bucket_token}.mp4"
      echo "Rendering turntable ${video_out}"
      "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_turntable_video \
        --left "${left_colored}" \
        --right "${right_colored}" \
        --output "${video_out}" \
        --left-title "${left_name} ${bucket_count} buckets" \
        --right-title "${right_name} ${bucket_count} buckets" \
        --frames 120 \
        --fps 24 \
        --width 1280 \
        --height 720 \
        --elevation 11 \
        --zoom 0.90 \
        --normalize independent
    fi
  done

  contact_sheet="${VISUAL_DIR}/${pair_slug}_partfield_bucket_contact_sheet.png"
  echo "Rendering contact sheet ${contact_sheet}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_bucket_contact_sheet \
    "${contact_args[@]}" \
    --output "${contact_sheet}" \
    --title "${left_name} -> ${right_name} PartField bucket comparison" \
    --left-title "${left_name}" \
    --right-title "${right_name}" \
    --azimuth -35 \
    --elevation 11 \
    --zoom 0.90 \
    --normalize independent
done

echo "END_TIME=$(date)"
echo "Feature files:"
find "${MESHUP_FEATURE_DIR}" -maxdepth 1 -type f -name 'part_feat_*_0*.npy' | sort
echo "Contact sheets:"
find "${VISUAL_DIR}" -maxdepth 1 -type f -name '*contact_sheet.png' | sort
