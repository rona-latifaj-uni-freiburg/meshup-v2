#!/bin/bash
#SBATCH --job-name=prep_pf_bears
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_bears_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_bears_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

PAIR_NAME=${PAIR_NAME:-bear_to_bear2}
MESH_DIR=${MESH_DIR:-${MESHUP_ROOT}/mesh_creator_for_meshup/sam3D/processed_meshes}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_sam3d_${PAIR_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_sam3d_${PAIR_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/sam3d_animals/${PAIR_NAME}}
SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/sam3d_animals/${PAIR_NAME}}
VISUAL_DIR=${VISUAL_DIR:-${SEGMENT_OUTPUT_ROOT}/visuals}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-6,7,8}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}

MESHES=(
  "${MESH_DIR}/bear.ply"
  "${MESH_DIR}/bear2.ply"
)

NAMES=(
  bear
  bear2
)

if [[ ! -d "${PARTFIELD_REPO}" || ! -f "${PARTFIELD_REPO}/partfield_inference.py" ]]; then
  echo "PartField checkout not found: ${PARTFIELD_REPO}"
  exit 1
fi

if [[ ! -f "${PARTFIELD_CKPT}" ]]; then
  echo "PartField checkpoint not found: ${PARTFIELD_CKPT}"
  exit 1
fi

if [[ "${#MESHES[@]}" -ne "${#NAMES[@]}" ]]; then
  echo "Internal error: MESHES/NAMES length mismatch."
  exit 1
fi

for mesh_path in "${MESHES[@]}"; do
  if [[ ! -f "${mesh_path}" ]]; then
    echo "Missing mesh: ${mesh_path}"
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
echo "Preparing PartField 6/7/8 bucket visuals for ${PAIR_NAME}"
echo "PARTFIELD_REPO=${PARTFIELD_REPO}"
echo "PARTFIELD_CKPT=${PARTFIELD_CKPT}"
echo "PARTFIELD_DATA_DIR=${PARTFIELD_DATA_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT}"
echo "VISUAL_DIR=${VISUAL_DIR}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST}"
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
      echo "${MESHUP_FEATURE_DIR}/$(basename "${candidate}")"
      return 0
    fi
  done
  echo "Missing PartField output for ${base} in ${PARTFIELD_OUTPUT_DIR}" >&2
  return 1
}

FEATURES=()
for name in "${NAMES[@]}"; do
  FEATURES+=("$(copy_feature "${name}")")
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
CONTACT_ARGS=()

for bucket_count in "${BUCKETS[@]}"; do
  bucket_token=$(printf "%02d" "${bucket_count}")
  scale_output_dir="${SEGMENT_OUTPUT_ROOT}/buckets_${bucket_token}"

  echo
  echo "Writing ${bucket_count}-bucket labels/colored PLYs to ${scale_output_dir}"
  python -m jobs_with_target_guidance.partfield_segment \
    "${SEG_ARGS[@]}" \
    --output-dir "${scale_output_dir}" \
    --n-buckets "${bucket_count}" \
    --position-weight "${POSITION_WEIGHT}" \
    --normal-weight "${NORMAL_WEIGHT}"

  left_colored="${scale_output_dir}/colored/bear_partfield_${bucket_token}_parts.ply"
  right_colored="${scale_output_dir}/colored/bear2_partfield_${bucket_token}_parts.ply"
  video_out="${VISUAL_DIR}/${PAIR_NAME}_partfield_buckets_${bucket_token}.mp4"

  echo "Rendering turntable ${video_out}"
  python -m jobs_with_target_guidance.render_partfield_turntable_video \
    --left "${left_colored}" \
    --right "${right_colored}" \
    --output "${video_out}" \
    --left-title "bear ${bucket_count} buckets" \
    --right-title "bear2 ${bucket_count} buckets" \
    --frames 120 \
    --fps 24 \
    --width 1280 \
    --height 720 \
    --elevation 11 \
    --zoom 0.90 \
    --normalize independent

  CONTACT_ARGS+=("--case" "${bucket_count}" "${left_colored}" "${right_colored}")
done

contact_sheet="${VISUAL_DIR}/${PAIR_NAME}_partfield_bucket_contact_sheet.png"
echo "Rendering contact sheet ${contact_sheet}"
python -m jobs_with_target_guidance.render_partfield_bucket_contact_sheet \
  "${CONTACT_ARGS[@]}" \
  --output "${contact_sheet}" \
  --title "bear -> bear2 PartField bucket comparison" \
  --left-title bear \
  --right-title bear2 \
  --azimuth -35 \
  --elevation 11 \
  --zoom 0.90 \
  --normalize independent

echo "END_TIME=$(date)"
echo "Feature files:"
printf "  %s\n" "${FEATURES[@]}"
echo "Segment outputs:"
for bucket_count in "${BUCKETS[@]}"; do
  bucket_token=$(printf "%02d" "${bucket_count}")
  echo "  ${SEGMENT_OUTPUT_ROOT}/buckets_${bucket_token}"
done
echo "Visual outputs:"
ls -lh "${VISUAL_DIR}"
