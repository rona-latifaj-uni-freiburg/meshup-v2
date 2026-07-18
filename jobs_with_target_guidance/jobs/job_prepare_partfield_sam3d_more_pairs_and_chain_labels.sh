#!/bin/bash
#SBATCH --job-name=prep_pf_more
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/prep_pf_more_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/prep_pf_more_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/logs

RUN_NAME=${RUN_NAME:-more_pairs_20260707}
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
PAIR_SEGMENT_ROOT=${PAIR_SEGMENT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/sam3d_animals/${RUN_NAME}}
PAIR_VISUAL_DIR=${PAIR_VISUAL_DIR:-${PAIR_SEGMENT_ROOT}/visuals}
CHAIN_RUN_NAME=${CHAIN_RUN_NAME:-panther_horse_giraffe_20260707}
CHAIN_SEGMENT_ROOT=${CHAIN_SEGMENT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/chain_animals/${CHAIN_RUN_NAME}}
CHAIN_VISUAL_DIR=${CHAIN_VISUAL_DIR:-${CHAIN_SEGMENT_ROOT}/visuals}
DENSE_FEATURE_DIR=${DENSE_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/densecorr3d_animals/densecorr3d_animals_20260706}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-5,6,7,8,9}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
REUSE_FEATURES=${REUSE_FEATURES:-1}

SAM3D_MESH_BASES=(
  goldie
  pug
  fox
  cat2
  fox3
  pig
  bear
  bear2
  horse2
  horse3
)

mesh_path() {
  local name="$1"
  case "${name}" in
    panther) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/071b8_toy_animals_017/simple_mesh.obj" ;;
    dense_giraffe|giraffe) echo "${MESHUP_ROOT}/external/DenseCorr3D/animals/34fb4_toy_animals_019/simple_mesh.obj" ;;
    *) echo "${MESH_DIR}/${name}.ply" ;;
  esac
}

feature_path() {
  local name="$1"
  local dir="${MESHUP_FEATURE_DIR}"
  case "${name}" in
    panther) dir="${DENSE_FEATURE_DIR}" ;;
    dense_giraffe|giraffe) dir="${DENSE_FEATURE_DIR}" ;;
  esac

  local feature_name="${name}"
  if [[ "${name}" == "dense_giraffe" ]]; then
    feature_name=giraffe
  fi

  local candidate
  for candidate in \
    "${dir}/part_feat_${feature_name}_0_batch.npy" \
    "${dir}/part_feat_${feature_name}_0.npy"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "Missing feature for ${name} in ${dir}" >&2
  return 1
}

all_sam3d_features_exist() {
  local base
  for base in "${SAM3D_MESH_BASES[@]}"; do
    if ! feature_path "${base}" >/dev/null; then
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

for base in "${SAM3D_MESH_BASES[@]}"; do
  if [[ ! -f "${MESH_DIR}/${base}.ply" ]]; then
    echo "Missing SAM3D mesh: ${MESH_DIR}/${base}.ply"
    exit 1
  fi
done

mkdir -p "${MESHUP_FEATURE_DIR}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}" "${CHAIN_SEGMENT_ROOT}" "${CHAIN_VISUAL_DIR}"

echo "======================================================"
echo "Preparing PartField visuals for SAM3D more pairs and chain labels"
echo "RUN_NAME=${RUN_NAME}"
echo "CHAIN_RUN_NAME=${CHAIN_RUN_NAME}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "PAIR_SEGMENT_ROOT=${PAIR_SEGMENT_ROOT}"
echo "CHAIN_SEGMENT_ROOT=${CHAIN_SEGMENT_ROOT}"
echo "N_BUCKETS_LIST=${N_BUCKETS_LIST}"
echo "START_TIME=$(date)"
echo "======================================================"

if [[ "${REUSE_FEATURES}" == "1" ]] && all_sam3d_features_exist; then
  echo "Reusing existing SAM3D PartField features in ${MESHUP_FEATURE_DIR}"
else
  if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
    source "${PARTFIELD_ACTIVATE}"
  elif [[ -n "${PARTFIELD_ENV}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${PARTFIELD_ENV}"
  fi

  PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
  mkdir -p "${PARTFIELD_DATA_DIR}"

  for base in "${SAM3D_MESH_BASES[@]}"; do
    mesh_file="${MESH_DIR}/${base}.ply"
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
  for base in "${SAM3D_MESH_BASES[@]}"; do
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
fi

cd "${MESHUP_ROOT}"

segment_pair_series() {
  local pair_slug="$1"
  local left_mesh_name="$2"
  local left_label_name="$3"
  local right_mesh_name="$4"
  local right_label_name="$5"
  local buckets_csv="$6"
  local output_root="$7"
  local visual_dir="$8"
  local title="$9"

  read -r -a buckets <<< "${buckets_csv//,/ }"
  local contact_args=()
  local bucket_count
  for bucket_count in "${buckets[@]}"; do
    local bucket_token
    bucket_token=$(printf "%02d" "${bucket_count}")
    local scale_output_dir="${output_root}/${pair_slug}/buckets_${bucket_token}"

    echo
    echo "Writing ${bucket_count}-bucket ${pair_slug} labels/colored PLYs to ${scale_output_dir}"
    "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
      --mesh "$(mesh_path "${left_mesh_name}")" \
      --feature "$(feature_path "${left_mesh_name}")" \
      --name "${left_label_name}" \
      --mesh "$(mesh_path "${right_mesh_name}")" \
      --feature "$(feature_path "${right_mesh_name}")" \
      --name "${right_label_name}" \
      --output-dir "${scale_output_dir}" \
      --n-buckets "${bucket_count}" \
      --position-weight "${POSITION_WEIGHT}" \
      --normal-weight "${NORMAL_WEIGHT}"

    contact_args+=(
      "--case" "${bucket_count}"
      "${scale_output_dir}/colored/${left_label_name}_partfield_${bucket_token}_parts.ply"
      "${scale_output_dir}/colored/${right_label_name}_partfield_${bucket_token}_parts.ply"
    )
  done

  local contact_sheet="${visual_dir}/${pair_slug}_partfield_bucket_contact_sheet.png"
  echo "Rendering contact sheet ${contact_sheet}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.render_partfield_bucket_contact_sheet \
    "${contact_args[@]}" \
    --output "${contact_sheet}" \
    --title "${title}" \
    --left-title "${left_label_name}" \
    --right-title "${right_label_name}" \
    --azimuth -35 \
    --elevation 11 \
    --zoom 0.90 \
    --normalize independent
}

segment_pair_series goldie_to_pug goldie goldie pug pug "${N_BUCKETS_LIST}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}" "goldie -> pug PartField bucket comparison"
segment_pair_series goldie_to_fox goldie goldie fox fox "${N_BUCKETS_LIST}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}" "goldie -> fox PartField bucket comparison"
segment_pair_series cat_to_fox3 cat2 cat fox3 fox3 "${N_BUCKETS_LIST}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}" "cat -> fox3 PartField bucket comparison"
segment_pair_series pig_to_pug pig pig pug pug "${N_BUCKETS_LIST}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}" "pig -> pug PartField bucket comparison"
segment_pair_series bear_to_bear2 bear bear bear2 bear2 "${N_BUCKETS_LIST}" "${PAIR_SEGMENT_ROOT}" "${PAIR_VISUAL_DIR}" "bear -> bear2 PartField bucket comparison"

segment_pair_series panther_to_horse2 panther panther horse2 horse2 5,6 "${CHAIN_SEGMENT_ROOT}" "${CHAIN_VISUAL_DIR}" "panther -> horse2 PartField bucket comparison"
segment_pair_series horse2_to_dense_giraffe horse2 horse2 dense_giraffe giraffe 8 "${CHAIN_SEGMENT_ROOT}" "${CHAIN_VISUAL_DIR}" "horse2 -> DenseCorr3D giraffe PartField bucket comparison"
segment_pair_series panther_to_horse3 panther panther horse3 horse3 5,6 "${CHAIN_SEGMENT_ROOT}" "${CHAIN_VISUAL_DIR}" "panther -> horse3 PartField bucket comparison"
segment_pair_series horse3_to_dense_giraffe horse3 horse3 dense_giraffe giraffe 9 "${CHAIN_SEGMENT_ROOT}" "${CHAIN_VISUAL_DIR}" "horse3 -> DenseCorr3D giraffe PartField bucket comparison"

echo "END_TIME=$(date)"
echo "Pair visuals:"
find "${PAIR_VISUAL_DIR}" -maxdepth 1 -type f | sort
echo "Chain visuals:"
find "${CHAIN_VISUAL_DIR}" -maxdepth 1 -type f | sort
