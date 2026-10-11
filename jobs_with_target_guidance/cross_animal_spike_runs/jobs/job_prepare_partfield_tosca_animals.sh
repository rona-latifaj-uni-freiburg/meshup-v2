#!/bin/bash
#SBATCH --job-name=prep_pf_tosca
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_pf_tosca_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_pf_tosca_%j.err

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"
mkdir -p jobs_with_target_guidance/cross_animal_spike_runs/logs

RUN_NAME=${RUN_NAME:-tosca_wolf_pilot_20260806}
PREPARED_ROOT=${PREPARED_ROOT:-${MESHUP_ROOT}/external/TOSCA/meshup_prepared}
MESH_NAMES_LIST=${MESH_NAMES_LIST:-wolf0,wolf1,wolf2}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_ENV=${PARTFIELD_ENV:-partfield}
PARTFIELD_ACTIVATE=${PARTFIELD_ACTIVATE:-}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_tosca_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_tosca_${RUN_NAME}_${SLURM_JOB_ID:-manual}}
PARTFIELD_N_POINT_PER_FACE=${PARTFIELD_N_POINT_PER_FACE:-500}
PARTFIELD_N_SAMPLE_EACH=${PARTFIELD_N_SAMPLE_EACH:-10000}
PARTFIELD_VAL_NUM_WORKERS=${PARTFIELD_VAL_NUM_WORKERS:-8}
MESHUP_PY=${MESHUP_PY:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python}
MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_features/tosca_animals/${RUN_NAME}}
SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT:-${MESHUP_ROOT}/jobs_with_target_guidance/partfield_segments/tosca_animals/${RUN_NAME}}
N_BUCKETS_LIST=${N_BUCKETS_LIST:-6,7,8}
POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}
REUSE_FEATURES=${REUSE_FEATURES:-1}

read -r -a MESH_NAMES <<< "${MESH_NAMES_LIST//,/ }"
read -r -a BUCKETS <<< "${N_BUCKETS_LIST//,/ }"

mesh_path() {
  echo "${PREPARED_ROOT}/objs/$1.obj"
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
    echo "Missing prepared TOSCA OBJ for ${name}: $(mesh_path "${name}")"
    exit 1
  fi
done

mkdir -p "${MESHUP_FEATURE_DIR}" "${SEGMENT_OUTPUT_ROOT}"

echo "======================================================"
echo "Preparing TOSCA PartField labels"
echo "RUN_NAME=${RUN_NAME}"
echo "MESH_NAMES_LIST=${MESH_NAMES_LIST}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_ROOT=${SEGMENT_OUTPUT_ROOT}"
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
  find "${PARTFIELD_DATA_DIR}" -maxdepth 1 -type f \( -name '*.obj' -o -name '*.mtl' -o -name '*.obj.mtl' \) -delete

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
  cd "${MESHUP_ROOT}"
fi

segment_args=()
for name in "${MESH_NAMES[@]}"; do
  segment_args+=(
    --mesh "$(mesh_path "${name}")"
    --feature "$(feature_path "${name}")"
    --name "${name}"
  )
done

for bucket_count in "${BUCKETS[@]}"; do
  bucket_token=$(printf "%02d" "${bucket_count}")
  scale_output_dir="${SEGMENT_OUTPUT_ROOT}/selected/buckets_${bucket_token}"
  echo
  echo "Writing ${bucket_count}-bucket labels to ${scale_output_dir}"
  "${MESHUP_PY}" -m jobs_with_target_guidance.partfield_segment \
    "${segment_args[@]}" \
    --output-dir "${scale_output_dir}" \
    --n-buckets "${bucket_count}" \
    --position-weight "${POSITION_WEIGHT}" \
    --normal-weight "${NORMAL_WEIGHT}"
done

echo "END_TIME=$(date)"
echo "Feature files:"
find "${MESHUP_FEATURE_DIR}" -maxdepth 1 -type f -name 'part_feat_*' | sort
echo "Label files:"
find "${SEGMENT_OUTPUT_ROOT}/selected" -path '*/labels/*_partfield_labels.npz' | sort
