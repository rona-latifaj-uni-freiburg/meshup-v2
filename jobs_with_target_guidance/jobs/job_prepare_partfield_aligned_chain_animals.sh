#!/bin/bash
#SBATCH --job-name=prep_pf_chain_x
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_chain_x_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_chain_x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

RUN_NAME=${RUN_NAME:-panther_horse_giraffe_face_x_20260707}
ALIGNED_MESH_DIR=${ALIGNED_MESH_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/aligned_meshes/chain_animals/${RUN_NAME}}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_chain_aligned_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_chain_aligned_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
FEATURE_DIR=${FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/chain_animals/${RUN_NAME}}
SEGMENT_ROOT=${SEGMENT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/chain_animals/${RUN_NAME}}
VISUAL_DIR=${VISUAL_DIR:-${SEGMENT_ROOT}/visuals}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
REUSE_FEATURES=${REUSE_FEATURES:-1}

MESH_NAMES=(panther horse2 horse3 giraffe)

mesh_path() {
  echo "${ALIGNED_MESH_DIR}/$1.obj"
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

for name in "${MESH_NAMES[@]}"; do
  if [[ ! -f "$(mesh_path "${name}")" ]]; then
    echo "Missing aligned mesh: $(mesh_path "${name}")"
    exit 1
  fi
done

mkdir -p "${FEATURE_DIR}" "${SEGMENT_ROOT}" "${VISUAL_DIR}"

echo "======================================================"
echo "Preparing aligned chain PartField visuals"
echo "RUN_NAME=${RUN_NAME}"
echo "ALIGNED_MESH_DIR=${ALIGNED_MESH_DIR}"
echo "FEATURE_DIR=${FEATURE_DIR}"
echo "SEGMENT_ROOT=${SEGMENT_ROOT}"
echo "START_TIME=$(date)"
echo "======================================================"

if [[ "${REUSE_FEATURES}" == "1" ]] && all_features_exist; then
  echo "Reusing existing aligned-chain PartField features in ${FEATURE_DIR}"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${PARTFIELD_ENV}"

  PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
  mkdir -p "${PARTFIELD_DATA_DIR}"

  for name in "${MESH_NAMES[@]}"; do
    cp -f "$(mesh_path "${name}")" "${PARTFIELD_DATA_DIR}/${name}.obj"
    echo "Copied ${PARTFIELD_DATA_DIR}/${name}.obj"
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
        cp -f "${candidate}" "${FEATURE_DIR}/"
        echo "Copied ${FEATURE_DIR}/$(basename "${candidate}")"
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

segment_pair_series() {
  local pair_slug="$1"
  local left_name="$2"
  local right_name="$3"
  local buckets_csv="$4"
  local title="$5"

  read -r -a buckets <<< "${buckets_csv//,/ }"
  local contact_args=()
  local bucket_count
  for bucket_count in "${buckets[@]}"; do
    local bucket_token
    bucket_token=$(printf "%02d" "${bucket_count}")
    local scale_output_dir="${SEGMENT_ROOT}/${pair_slug}/buckets_${bucket_token}"

    echo
    echo "Writing ${bucket_count}-bucket ${pair_slug} labels/colored PLYs to ${scale_output_dir}"
    "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
      --mesh "$(mesh_path "${left_name}")" \
      --feature "$(feature_path "${left_name}")" \
      --name "${left_name}" \
      --mesh "$(mesh_path "${right_name}")" \
      --feature "$(feature_path "${right_name}")" \
      --name "${right_name}" \
      --output-dir "${scale_output_dir}" \
      --n-buckets "${bucket_count}" \
      --position-weight "${POSITION_WEIGHT}" \
      --normal-weight "${NORMAL_WEIGHT}"

    contact_args+=(--case "${bucket_count}" \
      "${scale_output_dir}/colored/${left_name}_partfield_${bucket_token}_parts.ply" \
      "${scale_output_dir}/colored/${right_name}_partfield_${bucket_token}_parts.ply")
  done

  local visual_path="${VISUAL_DIR}/${pair_slug}_partfield_bucket_contact_sheet.png"
  echo "Rendering contact sheet ${visual_path}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_bucket_contact_sheet \
    "${contact_args[@]}" \
    --output "${visual_path}" \
    --title "${title}" \
    --left-title "${left_name}" \
    --right-title "${right_name}" \
    --normalize independent
}

segment_pair_series panther_to_horse2 panther horse2 "5,6" "aligned panther -> horse2 PartField bucket comparison"
segment_pair_series horse2_to_dense_giraffe horse2 giraffe "8" "aligned horse2 -> DenseCorr3D giraffe PartField bucket comparison"
segment_pair_series panther_to_horse3 panther horse3 "5,6" "aligned panther -> horse3 PartField bucket comparison"
segment_pair_series horse3_to_dense_giraffe horse3 giraffe "9" "aligned horse3 -> DenseCorr3D giraffe PartField bucket comparison"

echo "END_TIME=$(date)"
echo "Visuals:"
find "${VISUAL_DIR}" -maxdepth 1 -type f -name '*.png' | sort
