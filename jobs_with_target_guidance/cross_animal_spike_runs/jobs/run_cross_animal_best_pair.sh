#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: $0 PAIR_ID EPOCHS OUTPUT_ROOT RUN_TAG"
  echo "PAIR_ID: 0 dachshund->golden, 1 golden->dachshund, 2 dachshund->cat, 3 cat->dachshund, 4 bulldog->cat, 5 cat->bulldog, 6 bulldog->dachshund, 7 dachshund->bulldog"
  exit 2
fi

PAIR_ID="$1"
EPOCHS="$2"
OUTPUT_ROOT="$3"
RUN_TAG="$4"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
PARTFIELD_FEATURE_DIR=${PARTFIELD_FEATURE_DIR:-${BASE_DIR}/partfield/features/no_dino_animals}
PARTFIELD_LABEL_DIR=${PARTFIELD_LABEL_DIR:-${BASE_DIR}/partfield/segments/no_dino_animals_12/labels}

case "${PAIR_ID}" in
  0)
    export PAIR_SLUG=dachshund_to_golden_retriever
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_golden_retriever_no_dino_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_dachshund
    export TARGET_NAME=hound_to_golden_retriever_no_dino
    export PROMPT="a golden retriever dog"
    ;;
  1)
    export PAIR_SLUG=golden_retriever_to_dachshund
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_golden_retriever_no_dino_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_golden_retriever_no_dino
    export TARGET_NAME=hound_to_dachshund
    export PROMPT="a dachshund dog"
    ;;
  2)
    export PAIR_SLUG=dachshund_to_cat
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_dachshund
    export TARGET_NAME=hound_to_cat_no_dino
    export PROMPT="a cat"
    ;;
  3)
    export PAIR_SLUG=cat_to_dachshund
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_cat_no_dino
    export TARGET_NAME=hound_to_dachshund
    export PROMPT="a dachshund dog"
    ;;
  4)
    export PAIR_SLUG=bulldog_to_cat
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_bulldog_no_dino
    export TARGET_NAME=hound_to_cat_no_dino
    export PROMPT="a cat"
    ;;
  5)
    export PAIR_SLUG=cat_to_bulldog
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_cat_no_dino
    export TARGET_NAME=hound_to_bulldog_no_dino
    export PROMPT="a bulldog dog"
    ;;
  6)
    export PAIR_SLUG=bulldog_to_dachshund
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_bulldog_no_dino
    export TARGET_NAME=hound_to_dachshund
    export PROMPT="a dachshund dog"
    ;;
  7)
    export PAIR_SLUG=dachshund_to_bulldog
    export SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    export TARGET=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    export SOURCE_NAME=hound_to_dachshund
    export TARGET_NAME=hound_to_bulldog_no_dino
    export PROMPT="a bulldog dog"
    ;;
  *)
    echo "Unknown PAIR_ID=${PAIR_ID}. Expected 0..7."
    exit 2
    ;;
esac

export SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES:-${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_NAME}_0_batch.npy}
export TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES:-${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_NAME}_0_batch.npy}
export SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS:-${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz}
export TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS:-${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz}
export SEMANTIC_VERTEX_CORRESPONDENCE_CACHE=${SEMANTIC_VERTEX_CORRESPONDENCE_CACHE:-${BASE_DIR}/correspondences/${PAIR_SLUG}_partfield_semantic_vcorr.npz}

export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-0.0}
export TARGET_VERTEX_CORRESPONDENCE_WEIGHT=${TARGET_VERTEX_CORRESPONDENCE_WEIGHT:-20000.0}
export TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=${TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS:-0}
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-0.0}
export PARTFIELD_GUIDANCE_MODE=${PARTFIELD_GUIDANCE_MODE:-hard}
export PARTFIELD_LABELS_ALIGNED=${PARTFIELD_LABELS_ALIGNED:-0}
export PARTFIELD_HARD_WEIGHT=${PARTFIELD_HARD_WEIGHT:-1.0}
export PARTFIELD_SOFT_WEIGHT=${PARTFIELD_SOFT_WEIGHT:-0.0}
export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT:-0.55}
export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT:-0.80}
export PARTFIELD_TGT_TO_SRC_ROBUST_SCALE=${PARTFIELD_TGT_TO_SRC_ROBUST_SCALE:--1.0}
export PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=${PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT:-0.60}
export PARTFIELD_HARD_SEMANTIC_WEIGHT=${PARTFIELD_HARD_SEMANTIC_WEIGHT:-0.0}
export PARTFIELD_HARD_GEOMETRY_SIGMA=${PARTFIELD_HARD_GEOMETRY_SIGMA:-1.0}
export PARTFIELD_SEMANTIC_CONFIDENCE_MIN_SIMILARITY=${PARTFIELD_SEMANTIC_CONFIDENCE_MIN_SIMILARITY:--1.0}
export PARTFIELD_SEMANTIC_CONFIDENCE_MARGIN=${PARTFIELD_SEMANTIC_CONFIDENCE_MARGIN:-0.0}
export PARTFIELD_SEMANTIC_CONFIDENCE_FLOOR=${PARTFIELD_SEMANTIC_CONFIDENCE_FLOOR:-1.0}
export PARTFIELD_SEMANTIC_CONFIDENCE_POWER=${PARTFIELD_SEMANTIC_CONFIDENCE_POWER:-1.0}
export PARTFIELD_UNBALANCED_TRANSPORT_WEIGHT=${PARTFIELD_UNBALANCED_TRANSPORT_WEIGHT:-0.0}
export PARTFIELD_MOMENT_WEIGHT=${PARTFIELD_MOMENT_WEIGHT:-0.12}
export PARTFIELD_MOMENT_EXTENT_WEIGHT=${PARTFIELD_MOMENT_EXTENT_WEIGHT:-0.55}
export PARTFIELD_PROFILE_WEIGHT=${PARTFIELD_PROFILE_WEIGHT:-0.35}
export PARTFIELD_PROFILE_BINS=${PARTFIELD_PROFILE_BINS:-9}
export PARTFIELD_PROFILE_TRIM=${PARTFIELD_PROFILE_TRIM:-0.08}
export PARTFIELD_ANCHOR_WEIGHT=${PARTFIELD_ANCHOR_WEIGHT:-0.60}
export PARTFIELD_ANCHOR_GEOMETRY_SIGMA=${PARTFIELD_ANCHOR_GEOMETRY_SIGMA:-0.35}
export PARTFIELD_ANCHOR_SEMANTIC_WEIGHT=${PARTFIELD_ANCHOR_SEMANTIC_WEIGHT:-0.50}
export DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-vertex}
export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-0.0}
export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
export EDGE_STRETCH_THRESHOLD=${EDGE_STRETCH_THRESHOLD:-1.5}
export EDGE_STRETCH_MAX_WEIGHT=${EDGE_STRETCH_MAX_WEIGHT:-1.0}
export EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-0.0}
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.25}
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-1.0}
export ARTUR_SOFT_MATCH_SPACE=${ARTUR_SOFT_MATCH_SPACE:-hybrid}
export ARTUR_SOFT_POINTS=${ARTUR_SOFT_POINTS:-1024}
export ARTUR_SOFT_GEOMETRY_SIGMA=${ARTUR_SOFT_GEOMETRY_SIGMA:-0.5}
export ARTUR_SOFT_SEMANTIC_WEIGHT=${ARTUR_SOFT_SEMANTIC_WEIGHT:-1.0}
export ARTUR_SOFT_TEMPERATURE=${ARTUR_SOFT_TEMPERATURE:-0.1}
export ENABLE_SOURCE_DINO_LOSS=0
export ENABLE_TARGET_DINO_GUIDANCE=0
export TARGET_CHAMFER_WARMUP_EPOCHS=${TARGET_CHAMFER_WARMUP_EPOCHS:-0}
export PARTFIELD_CHAMFER_WARMUP_EPOCHS=${PARTFIELD_CHAMFER_WARMUP_EPOCHS:-0}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-250}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-250}
export VARIANT_SUFFIX=${VARIANT_SUFFIX:-topomatch_vcorr20000_vertex}

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  1 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

submit_clean_pair_job() (
  unset PAIR_SLUG
  unset SOURCE
  unset TARGET
  unset SOURCE_NAME
  unset TARGET_NAME
  unset PROMPT
  unset SOURCE_PARTFIELD_FEATURES
  unset TARGET_PARTFIELD_FEATURES
  unset SOURCE_PARTFIELD_LABELS
  unset TARGET_PARTFIELD_LABELS
  sbatch --parsable "${BASE_DIR}/jobs/job_dev_h100_cross_animal_best_single.sh" "$1"
)

if [[ -n "${SLURM_JOB_ID:-}" && "${CHAIN_NEXT_PAIRS:-0}" == "1" ]]; then
  mkdir -p "${BASE_DIR}/reports"
  if (( PAIR_ID < 7 )); then
    next_pair=$((PAIR_ID + 1))
    next_job=$(submit_clean_pair_job "${next_pair}")
    {
      echo "finished_pair=${PAIR_ID}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "submitted_next_pair=${next_pair}"
      echo "next_job=${next_job}"
      echo "submitted_at=$(date)"
    } > "${BASE_DIR}/reports/auto_submitted_after_pair${PAIR_ID}.txt"
  else
    analysis_job=$(sbatch --parsable "${BASE_DIR}/jobs/job_analyze_cross_animal_outputs.sh")
    {
      echo "finished_pair=${PAIR_ID}"
      echo "finished_job=${SLURM_JOB_ID}"
      echo "analysis_job=${analysis_job}"
      echo "submitted_at=$(date)"
    } > "${BASE_DIR}/reports/auto_submitted_analysis_after_pair${PAIR_ID}.txt"
  fi
elif [[ "${CHAIN_REMAINING_AFTER_PAIR3:-0}" == "1" && "${PAIR_ID}" == "3" && -n "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${BASE_DIR}/reports"
  echo "PAIR_ID=3 finished in job ${SLURM_JOB_ID}; submitting pair 4 with pair chaining enabled."
  export CHAIN_NEXT_PAIRS=1
  pair4_job=$(submit_clean_pair_job 4)
  {
    echo "submitted_after_pair3_job=${SLURM_JOB_ID}"
    echo "bulldog_to_cat_job=${pair4_job}"
    echo "submitted_at=$(date)"
  } > "${BASE_DIR}/reports/auto_submitted_remaining_jobs.txt"
fi
