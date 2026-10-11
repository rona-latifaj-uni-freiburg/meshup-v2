#!/bin/bash
#SBATCH --job-name=prep_dc3d_nonanimal_pf
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --array=0-22%8
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_densecorr3d_nonanimal_pf_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_densecorr3d_nonanimal_pf_%A_%a.err

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"

BASE_DIR=${BASE_DIR:-jobs_with_target_guidance/cross_animal_spike_runs}
RUN_NAME=${RUN_NAME:-densecorr3d_nonanimals_test_20260806}
MANIFEST=${MANIFEST:-${BASE_DIR}/jobs/densecorr3d_nonanimals_test_manifest.tsv}
CATEGORY_LIST=${CATEGORY_LIST:-${BASE_DIR}/jobs/densecorr3d_nonanimals_test_categories.txt}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
MESHUP_FEATURE_ROOT=${MESHUP_FEATURE_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/densecorr3d_categories/${RUN_NAME}}
SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/densecorr3d_categories/${RUN_NAME}}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-4,5,6,7,8}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
REUSE_FEATURES=${REUSE_FEATURES:-1}

TASK_ID=${1:-${SLURM_ARRAY_TASK_ID:-}}
if [[ -z "${TASK_ID}" ]]; then
  echo "Missing task id. Submit as an array job or pass a zero-based category task id."
  exit 2
fi

if [[ ! -f "${MANIFEST}" || ! -f "${CATEGORY_LIST}" ]]; then
  echo "Missing manifest/category list. Generate them first:"
  echo "  python ${BASE_DIR}/jobs/generate_densecorr3d_category_partfield_specs.py"
  exit 2
fi

CATEGORY=$(sed -n "$((TASK_ID + 1))p" "${CATEGORY_LIST}")
if [[ -z "${CATEGORY}" ]]; then
  echo "No category found for task id ${TASK_ID} in ${CATEGORY_LIST}"
  exit 2
fi

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

mkdir -p "${BASE_DIR}/logs" "${MESHUP_FEATURE_ROOT}/${CATEGORY}" "${SEGMENT_OUTPUT_ROOT}/${CATEGORY}"

echo "Preparing DenseCorr3D non-animal PartField assets"
echo "RUN_NAME=${RUN_NAME}"
echo "CATEGORY=${CATEGORY}"
echo "TASK_ID=${TASK_ID}"
echo "MANIFEST=${MANIFEST}"
echo "FEATURE_DIR=${MESHUP_FEATURE_ROOT}/${CATEGORY}"
echo "SEGMENT_DIR=${SEGMENT_OUTPUT_ROOT}/${CATEGORY}"
echo "START_TIME=$(date)"

all_features_exist=1
while IFS=$'\t' read -r category object_id name mesh groups feature feature_alt segment_root prompt; do
  [[ "${category}" == "category" ]] && continue
  [[ "${category}" != "${CATEGORY}" ]] && continue
  if [[ ! -f "${MESHUP_FEATURE_ROOT}/${CATEGORY}/part_feat_${name}_0_batch.npy" && ! -f "${MESHUP_FEATURE_ROOT}/${CATEGORY}/part_feat_${name}_0.npy" ]]; then
    all_features_exist=0
  fi
done < "${MANIFEST}"

if [[ "${REUSE_FEATURES}" == "1" && "${all_features_exist}" == "1" ]]; then
  echo "Reusing existing PartField features for ${CATEGORY}"
else
  if [[ -n "${PARTFIELD_ACTIVATE}" ]]; then
    source "${PARTFIELD_ACTIVATE}"
  elif [[ -n "${PARTFIELD_ENV}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${PARTFIELD_ENV}"
  fi

  PARTFIELD_DATA_SUBDIR="data/meshup_densecorr3d_${RUN_NAME}_${CATEGORY}_${SLURM_JOB_ID:-manual}_${TASK_ID}"
  PARTFIELD_RESULT_NAME="partfield_features/meshup_densecorr3d_${RUN_NAME}_${CATEGORY}_${SLURM_JOB_ID:-manual}_${TASK_ID}"
  PARTFIELD_DATA_DIR="${PARTFIELD_REPO}/${PARTFIELD_DATA_SUBDIR}"
  PARTFIELD_OUTPUT_DIR="${PARTFIELD_REPO}/exp_results/${PARTFIELD_RESULT_NAME}"
  mkdir -p "${PARTFIELD_DATA_DIR}"
  find "${PARTFIELD_DATA_DIR}" -maxdepth 1 -type f \( -name '*.obj' -o -name '*.mtl' -o -name '*.obj.mtl' \) -delete

  while IFS=$'\t' read -r category object_id name mesh groups feature feature_alt segment_root prompt; do
    [[ "${category}" == "category" ]] && continue
    [[ "${category}" != "${CATEGORY}" ]] && continue
    if [[ ! -f "${mesh}" ]]; then
      echo "Missing mesh: ${mesh}"
      exit 1
    fi
    cp -f "${mesh}" "${PARTFIELD_DATA_DIR}/${name}.obj"
  done < "${MANIFEST}"

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

  cd "${MESHUP_ROOT}"
  while IFS=$'\t' read -r category object_id name mesh groups feature feature_alt segment_root prompt; do
    [[ "${category}" == "category" ]] && continue
    [[ "${category}" != "${CATEGORY}" ]] && continue
    copied=0
    for candidate in \
      "${PARTFIELD_OUTPUT_DIR}/part_feat_${name}_0_batch.npy" \
      "${PARTFIELD_OUTPUT_DIR}/part_feat_${name}_0.npy"; do
      if [[ -f "${candidate}" ]]; then
        cp -f "${candidate}" "${MESHUP_FEATURE_ROOT}/${CATEGORY}/"
        copied=1
        break
      fi
    done
    if [[ "${copied}" != "1" ]]; then
      echo "Missing PartField output for ${name} in ${PARTFIELD_OUTPUT_DIR}"
      exit 1
    fi
  done < "${MANIFEST}"
fi

read -r -a BUCKETS <<< "${N_BUCKETS_LIST//,/ }"
for bucket_count in "${BUCKETS[@]}"; do
  bucket_token=$(printf "%02d" "${bucket_count}")
  scale_output_dir="${SEGMENT_OUTPUT_ROOT}/${CATEGORY}/buckets_${bucket_token}"
  segment_args=()
  while IFS=$'\t' read -r category object_id name mesh groups feature feature_alt segment_root prompt; do
    [[ "${category}" == "category" ]] && continue
    [[ "${category}" != "${CATEGORY}" ]] && continue
    feature_path="${MESHUP_FEATURE_ROOT}/${CATEGORY}/part_feat_${name}_0_batch.npy"
    if [[ ! -f "${feature_path}" ]]; then
      feature_path="${MESHUP_FEATURE_ROOT}/${CATEGORY}/part_feat_${name}_0.npy"
    fi
    if [[ ! -f "${feature_path}" ]]; then
      echo "Missing feature for ${name}: ${feature_path}"
      exit 1
    fi
    segment_args+=(--mesh "${mesh}" --feature "${feature_path}" --name "${name}")
  done < "${MANIFEST}"

  echo "Writing ${bucket_count}-bucket labels for ${CATEGORY} to ${scale_output_dir}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
    "${segment_args[@]}" \
    --output-dir "${scale_output_dir}" \
    --n-buckets "${bucket_count}" \
    --position-weight "${POSITION_WEIGHT}" \
    --normal-weight "${NORMAL_WEIGHT}"
done

echo "END_TIME=$(date)"
