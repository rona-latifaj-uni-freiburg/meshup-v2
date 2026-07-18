#!/bin/bash
#SBATCH --job-name=oracle_cat_dach
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/oracle_semantic_buckets/logs/oracle_cat_dach_%j.out
#SBATCH --error=jobs_with_target_guidance/oracle_semantic_buckets/logs/oracle_cat_dach_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=./jobs_with_target_guidance/oracle_semantic_buckets
LABEL_DIR=${BASE_DIR}/cat_to_dachshund_shared_topology/labels
mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/outputs/dev_oracle_semantic"

export PAIR_SLUG=cat_to_dachshund
export SOURCE=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
export TARGET=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
export SOURCE_NAME=hound_to_cat_no_dino
export TARGET_NAME=hound_to_dachshund
export PROMPT="a dachshund dog"

export SOURCE_PARTFIELD_LABELS=${LABEL_DIR}/source_cat_oracle_semantic_labels.npz
export TARGET_PARTFIELD_LABELS=${LABEL_DIR}/target_dachshund_oracle_semantic_labels.npz
export PARTFIELD_USE_FEATURES=0
export PARTFIELD_LABELS_ALIGNED=1
export PARTFIELD_N_BUCKETS=10
export PARTFIELD_MIN_POINTS=12

export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=0.0
export TARGET_VERTEX_CORRESPONDENCE_WEIGHT=0.0
export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=0.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=8000.0
export PARTFIELD_GUIDANCE_MODE=hard
export PARTFIELD_HARD_WEIGHT=1.0
export PARTFIELD_SOFT_WEIGHT=0.0
export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.35
export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0
export PARTFIELD_TGT_TO_SRC_ROBUST_SCALE=0.0
export PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=1.0

export DEFORMATION_PARAMETERIZATION=jacobian
export JACOBIAN_REG_WEIGHT=0.0
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000.0
export EDGE_STRETCH_WEIGHT=0.0
export EDGE_DISPLACEMENT_JUMP_WEIGHT=500.0
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.2
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.0

export LOG_INTERVAL_IM=50
export SAVE_RENDERS_INTERVAL=50
export VARIANT_SUFFIX=oracle_semantic10_asym035_jneighbor1000_jump500

EPOCHS=${EPOCHS:-250}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/dev_oracle_semantic}
RUN_TAG=${RUN_TAG:-oracle_semantic_dev_h100}

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  1 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
