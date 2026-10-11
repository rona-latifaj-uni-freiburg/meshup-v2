#!/bin/bash
#SBATCH --job-name=tosca_vcorr
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --array=0-22
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/tosca_vcorr_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/tosca_vcorr_%A_%a.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

if [[ -z "${TASK_IDS:-}" && -n "${TASK_IDS_RANGE:-}" ]]; then
  IFS='-' read -r TASK_RANGE_START TASK_RANGE_END <<< "${TASK_IDS_RANGE}"
  if [[ -z "${TASK_RANGE_START:-}" || -z "${TASK_RANGE_END:-}" ]]; then
    echo "Malformed TASK_IDS_RANGE=${TASK_IDS_RANGE}; expected START-END"
    exit 2
  fi
  TASK_IDS=""
  for ((task_id = TASK_RANGE_START; task_id <= TASK_RANGE_END; task_id++)); do
    TASK_IDS+="${task_id} "
  done
fi

if [[ -n "${TASK_IDS:-}" && -z "${TOSCA_VCORR_SINGLE_TASK:-}" ]]; then
  TASK_IDS="${TASK_IDS//,/ }"
  echo "Running bundled TOSCA correspondence tasks: ${TASK_IDS}"
  for task_id in ${TASK_IDS}; do
    echo "======================================================"
    echo "Starting bundled task ${task_id}"
    TOSCA_VCORR_SINGLE_TASK=1 TASK_IDS= "$0" "${task_id}"
    echo "Finished bundled task ${task_id}"
  done
  exit 0
fi

SPEC_FILE=${SPEC_FILE:-jobs_with_target_guidance/cross_animal_spike_runs/jobs/tosca_correspondence_guided_specs.tsv}
TASK_ID=${1:-${SLURM_ARRAY_TASK_ID:-}}
EPOCHS=${EPOCHS:-2000}

if [[ -z "${TASK_ID}" ]]; then
  echo "Missing task id. Submit as an array job or pass a zero-based task id."
  exit 2
fi

if ! LINE=$(awk -F '\t' -v task_id="${TASK_ID}" '
  BEGIN { n = 0; found = 0 }
  /^[[:space:]]*#/ || NF == 0 { next }
  {
    if (n == task_id) {
      print
      found = 1
      exit
    }
    n++
  }
  END { if (!found) exit 1 }
' "${SPEC_FILE}"); then
  echo "No spec row found for task id ${TASK_ID} in ${SPEC_FILE}"
  exit 2
fi

IFS=$'\t' read -r \
  PAIR_SLUG \
  SOURCE \
  TARGET \
  SOURCE_NAME \
  TARGET_NAME \
  SOURCE_PARTFIELD_LABELS \
  TARGET_PARTFIELD_LABELS \
  PROMPT <<< "${LINE}"

if [[ -z "${PAIR_SLUG}" || -z "${SOURCE}" || -z "${TARGET}" || -z "${PROMPT}" ]]; then
  echo "Malformed spec row for task id ${TASK_ID}: ${LINE}"
  exit 2
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-./jobs_with_target_guidance/cross_animal_spike_runs/outputs/tosca_correspondence_guided}
RUN_TAG=${RUN_TAG:-tosca_vcorr_2000}

export SOURCE TARGET SOURCE_NAME TARGET_NAME PAIR_SLUG PROMPT
export SOURCE_PARTFIELD_LABELS TARGET_PARTFIELD_LABELS

export PARTFIELD_USE_FEATURES=0
export PARTFIELD_N_BUCKETS=1
export PARTFIELD_LABELS_ALIGNED=1
export PARTFIELD_GUIDANCE_MODE=hard
export PARTFIELD_HARD_WEIGHT=0.0
export PARTFIELD_SOFT_WEIGHT=0.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=0.0
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=0.0

export TARGET_VERTEX_CORRESPONDENCE_WEIGHT=${TARGET_VERTEX_CORRESPONDENCE_WEIGHT:-20000.0}
export TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=${TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS:-0}
export TARGET_VERTEX_CORRESPONDENCE_REQUIRE_SAME_FACES=0
export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=0.0
export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=0

export DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-jacobian}
export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-100.0}
export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
export JACOBIAN_OUTLIER_POWER=4.0
export DEFORMATION_GRAD_CLIP_NORM=0.0
export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
export EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-50.0}
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.2
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.0

export TARGET_CHAMFER_WARMUP_EPOCHS=0
export PARTFIELD_CHAMFER_WARMUP_EPOCHS=0
export ENABLE_SOURCE_DINO_LOSS=0
export ENABLE_TARGET_DINO_GUIDANCE=0
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-250}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-500}
export SAVE_PCA_VISUALIZATION=0
export EXTRA_LOG_EPOCHS="${EXTRA_LOG_EPOCHS:-0 1 10 100 500 1000 2000}"
export VARIANT_SUFFIX="${PAIR_SLUG}_tosca_identity_vcorr_w20000_jacns100_jump50_${EPOCHS}ep"

echo "Running TOSCA correspondence-guided deformation task ${TASK_ID}: ${PAIR_SLUG}"
echo "SOURCE=${SOURCE}"
echo "TARGET=${TARGET}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EPOCHS=${EPOCHS}"

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  1 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
