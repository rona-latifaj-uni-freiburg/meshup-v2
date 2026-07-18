#!/bin/bash
#SBATCH --job-name=h100_dm_pair
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:30:00
#SBATCH --output=jobs_with_target_guidance/densematcher_runs/logs/h100_dm_pair_%j.out
#SBATCH --error=jobs_with_target_guidance/densematcher_runs/logs/h100_dm_pair_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: sbatch $0 VARIANT SOURCE_OBJECT TARGET_OBJECT [EPOCHS]"
  echo "Example: sbatch $0 equal4996 2d6b3_toy_animals_009 34fb4_toy_animals_019 4000"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

VARIANT="$1"
SOURCE_OBJECT="$2"
TARGET_OBJECT="$3"
EPOCHS="${4:-${EPOCHS:-4000}}"

if [[ ! "${VARIANT}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "VARIANT contains unsupported characters: ${VARIANT}"
  exit 2
fi

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/densematcher_runs}
PREPARED_DIR=${PREPARED_DIR:-${BASE_DIR}/prepared/elephant_giraffe}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/elephant_to_giraffe}
RUN_TAG=${RUN_TAG:-h100_densecorr3d}

SOURCE="${PREPARED_DIR}/meshes/${SOURCE_OBJECT}_densecorr3d_${VARIANT}.obj"
TARGET="${PREPARED_DIR}/meshes/${TARGET_OBJECT}_densecorr3d_${VARIANT}.obj"
SOURCE_PARTFIELD_LABELS="${PREPARED_DIR}/labels/${SOURCE_OBJECT}_densecorr3d_${VARIANT}_labels.npz"
TARGET_PARTFIELD_LABELS="${PREPARED_DIR}/labels/${TARGET_OBJECT}_densecorr3d_${VARIANT}_labels.npz"

for path in "${SOURCE}" "${TARGET}" "${SOURCE_PARTFIELD_LABELS}" "${TARGET_PARTFIELD_LABELS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing prepared DenseCorr3D input: ${path}"
    echo "Run jobs_with_target_guidance/densecorr3d_prepare_mesh_variants.py first."
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

export SOURCE
export TARGET
export SOURCE_NAME="${SOURCE_OBJECT}_${VARIANT}"
export TARGET_NAME="${TARGET_OBJECT}_${VARIANT}"
export PAIR_SLUG="${SOURCE_OBJECT}_to_${TARGET_OBJECT}_${VARIANT}"
export PROMPT=${PROMPT:-"an elephant deformed into a giraffe"}
export SOURCE_PARTFIELD_LABELS
export TARGET_PARTFIELD_LABELS
export PARTFIELD_USE_FEATURES=0
export PARTFIELD_LABELS_ALIGNED=${PARTFIELD_LABELS_ALIGNED:-1}
export PARTFIELD_N_BUCKETS=${PARTFIELD_N_BUCKETS:-8}
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-8000.0}
export PARTFIELD_GUIDANCE_MODE=${PARTFIELD_GUIDANCE_MODE:-hard}
export PARTFIELD_HARD_WEIGHT=${PARTFIELD_HARD_WEIGHT:-1.0}
export PARTFIELD_SOFT_WEIGHT=${PARTFIELD_SOFT_WEIGHT:-0.0}
export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT:-0.35}
export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT:-1.0}
export PARTFIELD_TGT_TO_SRC_ROBUST_SCALE=${PARTFIELD_TGT_TO_SRC_ROBUST_SCALE:-0.0}
export PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=${PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT:-1.0}
export PARTFIELD_MIN_POINTS=${PARTFIELD_MIN_POINTS:-8}
export DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-jacobian}
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
export EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-500.0}
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
export TARGET_VERTEX_CORRESPONDENCE_WEIGHT=${TARGET_VERTEX_CORRESPONDENCE_WEIGHT:-0.0}
export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=${TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT:-0.0}
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-0.0}
export ENABLE_SOURCE_DINO_LOSS=${ENABLE_SOURCE_DINO_LOSS:-0}
export ENABLE_TARGET_DINO_GUIDANCE=${ENABLE_TARGET_DINO_GUIDANCE:-0}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-500}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-500}
export VARIANT_SUFFIX=${VARIANT_SUFFIX:-densecorr3d_${VARIANT}_groups}

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  1 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
