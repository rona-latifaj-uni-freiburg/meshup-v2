#!/bin/bash
#SBATCH --job-name=llm_horse_giraffe
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:30:00
#SBATCH --output=jobs_with_target_guidance/full_pipeline_runs/horse2_to_giraffe_llm_bucket_20260819/logs/full_pipeline_%j.out
#SBATCH --error=jobs_with_target_guidance/full_pipeline_runs/horse2_to_giraffe_llm_bucket_20260819/logs/full_pipeline_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
source ./activate_meshup_new.sh

RUN_ROOT=${RUN_ROOT:-jobs_with_target_guidance/full_pipeline_runs/horse2_to_giraffe_llm_bucket_20260819}
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/decisions" "${RUN_ROOT}/partfield_segments" "${RUN_ROOT}/outputs" "${RUN_ROOT}/videos"

SOURCE=${SOURCE:-jobs_with_target_guidance/aligned_meshes/chain_animals/panther_horse_giraffe_face_x_20260707/horse2.obj}
TARGET=${TARGET:-jobs_with_target_guidance/aligned_meshes/chain_animals/panther_horse_giraffe_face_x_20260707/giraffe.obj}
SOURCE_NAME=${SOURCE_NAME:-horse2}
TARGET_NAME=${TARGET_NAME:-giraffe}
SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES:-jobs_with_target_guidance/partfield_features/chain_animals/panther_horse_giraffe_face_x_20260707/part_feat_horse2_0_batch.npy}
TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES:-jobs_with_target_guidance/partfield_features/chain_animals/panther_horse_giraffe_face_x_20260707/part_feat_giraffe_0_batch.npy}
PAIR_SLUG=${PAIR_SLUG:-horse2_to_giraffe}
PROMPT=${PROMPT:-"a horse deformed into a giraffe"}
EPOCHS=${EPOCHS:-6000}
RUN_TAG=${RUN_TAG:-llm_bucket_full_pipeline}
JOB_STEM=${SLURM_JOB_ID:-manual}

for path in "${SOURCE}" "${TARGET}" "${SOURCE_PARTFIELD_FEATURES}" "${TARGET_PARTFIELD_FEATURES}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}"
    exit 1
  fi
done

SELECTOR_DECISION=${SELECTOR_DECISION:-${RUN_ROOT}/decisions/${PAIR_SLUG}_bucket_decision_${JOB_STEM}.json}
PARTFIELD_BUCKET_SELECTOR_PROVIDER=${PARTFIELD_BUCKET_SELECTOR_PROVIDER:-auto}
PARTFIELD_BUCKET_SELECTOR_CANDIDATES=${PARTFIELD_BUCKET_SELECTOR_CANDIDATES:-4,5,6,7,8,9,10}
PARTFIELD_BUCKET_SELECTOR_REQUEST=${PARTFIELD_BUCKET_SELECTOR_REQUEST:-"Choose one fixed PartField bucket count for a full horse-to-giraffe pipeline. Prefer enough parts to represent legs, torso, head, and the giraffe neck without over-fragmenting the meshes."}

selector_output=$(
  python -m jobs_with_target_guidance.partfield_bucket_llm \
    --source-mesh "${SOURCE}" \
    --target-mesh "${TARGET}" \
    --prompt "${PROMPT}" \
    --user-request "${PARTFIELD_BUCKET_SELECTOR_REQUEST}" \
    --mode fixed \
    --candidate-buckets "${PARTFIELD_BUCKET_SELECTOR_CANDIDATES}" \
    --provider "${PARTFIELD_BUCKET_SELECTOR_PROVIDER}" \
    --output "${SELECTOR_DECISION}" \
    --print-key shell
)

while IFS='=' read -r selector_key selector_value; do
  case "${selector_key}" in
    PARTFIELD_N_BUCKETS) PARTFIELD_N_BUCKETS="${selector_value}" ;;
    PARTFIELD_BUCKETS) PARTFIELD_BUCKETS="${selector_value}" ;;
    PARTFIELD_BUCKET_SELECTOR_PROVIDER) RESOLVED_PARTFIELD_BUCKET_SELECTOR_PROVIDER="${selector_value}" ;;
    PARTFIELD_BUCKET_SELECTOR_CONFIDENCE) PARTFIELD_BUCKET_SELECTOR_CONFIDENCE="${selector_value}" ;;
  esac
done <<< "${selector_output}"

if [[ -z "${PARTFIELD_N_BUCKETS:-}" ]]; then
  echo "Bucket selector did not return PARTFIELD_N_BUCKETS."
  echo "${selector_output}"
  exit 1
fi

echo "Selected PartField buckets: ${PARTFIELD_N_BUCKETS}"
echo "Selector provider: ${RESOLVED_PARTFIELD_BUCKET_SELECTOR_PROVIDER:-${PARTFIELD_BUCKET_SELECTOR_PROVIDER}}"
echo "Selector confidence: ${PARTFIELD_BUCKET_SELECTOR_CONFIDENCE:-unknown}"
echo "Selector decision: ${SELECTOR_DECISION}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; stopping before PartField segmentation and deformation."
  echo "SOURCE=${SOURCE}"
  echo "TARGET=${TARGET}"
  echo "SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES}"
  echo "TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES}"
  echo "RUN_ROOT=${RUN_ROOT}"
  exit 0
fi

SEGMENT_DIR=${SEGMENT_DIR:-${RUN_ROOT}/partfield_segments/${PAIR_SLUG}_buckets_${PARTFIELD_N_BUCKETS}_${JOB_STEM}}
python jobs_with_target_guidance/partfield_segment.py \
  --mesh "${SOURCE}" \
  --mesh "${TARGET}" \
  --feature "${SOURCE_PARTFIELD_FEATURES}" \
  --feature "${TARGET_PARTFIELD_FEATURES}" \
  --name "${SOURCE_NAME}" \
  --name "${TARGET_NAME}" \
  --output-dir "${SEGMENT_DIR}" \
  --n-buckets "${PARTFIELD_N_BUCKETS}" \
  --feature-mode auto \
  --position-weight "${PARTFIELD_SEGMENT_POSITION_WEIGHT:-0.05}" \
  --normal-weight "${PARTFIELD_SEGMENT_NORMAL_WEIGHT:-0.0}" \
  --random-state "${PARTFIELD_SEGMENT_RANDOM_STATE:-42}"

export SOURCE
export TARGET
export SOURCE_NAME
export TARGET_NAME
export PAIR_SLUG
export PROMPT
export PARTFIELD_N_BUCKETS
export PARTFIELD_USE_FEATURES=1
export SOURCE_PARTFIELD_FEATURES
export TARGET_PARTFIELD_FEATURES
export SOURCE_PARTFIELD_LABELS="${SEGMENT_DIR}/labels/${SOURCE_NAME}_partfield_labels.npz"
export TARGET_PARTFIELD_LABELS="${SEGMENT_DIR}/labels/${TARGET_NAME}_partfield_labels.npz"
export PARTFIELD_LABELS_ALIGNED=1
export PARTFIELD_MIN_POINTS=${PARTFIELD_MIN_POINTS:-12}

export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-8000.0}
export PARTFIELD_GUIDANCE_MODE=${PARTFIELD_GUIDANCE_MODE:-hard}
export PARTFIELD_HARD_WEIGHT=${PARTFIELD_HARD_WEIGHT:-1.0}
export PARTFIELD_SOFT_WEIGHT=${PARTFIELD_SOFT_WEIGHT:-0.0}
export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT:-0.35}
export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT:-1.0}
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-0.0}
export ENABLE_SOURCE_DINO_LOSS=${ENABLE_SOURCE_DINO_LOSS:-0}
export ENABLE_TARGET_DINO_GUIDANCE=${ENABLE_TARGET_DINO_GUIDANCE:-0}
export DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-jacobian}
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
export EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-500.0}
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-500}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-500}
export EXTRA_LOG_EPOCHS="${EXTRA_LOG_EPOCHS:-1 6000}"
export SAVE_PCA_VISUALIZATION=0
export VARIANT_SUFFIX="partfield_llm_bucket_${PARTFIELD_N_BUCKETS}"

OUTPUT_ROOT="${RUN_ROOT}/outputs"
RUN_DIR="${OUTPUT_ROOT}/${PAIR_SLUG}_${RUN_TAG}_hard_partfield_chamfer_only_${VARIANT_SUFFIX}_${EPOCHS}ep_${JOB_STEM}"

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  1 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

VIDEO_OUT="${RUN_ROOT}/videos/${PAIR_SLUG}_start_to_result_buckets_${PARTFIELD_N_BUCKETS}_${JOB_STEM}.mp4"
python jobs_with_target_guidance/render_partfield_turntable_video.py \
  --left "${RUN_DIR}/colored_meshes/mesh_initial.ply" \
  --right "${RUN_DIR}/colored_meshes/mesh_final_position.ply" \
  --output "${VIDEO_OUT}" \
  --left-title "Start" \
  --right-title "Result" \
  --frames "${VIDEO_FRAMES:-144}" \
  --fps "${VIDEO_FPS:-24}" \
  --width "${VIDEO_WIDTH:-1280}" \
  --height "${VIDEO_HEIGHT:-720}" \
  --elevation "${VIDEO_ELEVATION:-11}" \
  --zoom "${VIDEO_ZOOM:-0.94}" \
  --normalize independent

echo "======================================================"
echo "Full pipeline complete"
echo "RUN_DIR=${RUN_DIR}"
echo "SEGMENT_DIR=${SEGMENT_DIR}"
echo "SELECTOR_DECISION=${SELECTOR_DECISION}"
echo "VIDEO_OUT=${VIDEO_OUT}"
echo "======================================================"
