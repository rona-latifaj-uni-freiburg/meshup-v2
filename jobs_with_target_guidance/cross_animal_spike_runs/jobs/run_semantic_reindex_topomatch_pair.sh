#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 PAIR_ID [DEFORMED_RUN_DIR]"
  echo "PAIR_ID: 4 bulldog->cat, 5 cat->bulldog, 6 bulldog->dachshund, 7 dachshund->bulldog"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
source ./activate_meshup_new.sh

PAIR_ID="$1"
DEFORMED_RUN_DIR="${2:-}"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
PARTFIELD_FEATURE_DIR=${PARTFIELD_FEATURE_DIR:-${BASE_DIR}/partfield/features/no_dino_animals}
PARTFIELD_LABEL_DIR=${PARTFIELD_LABEL_DIR:-${BASE_DIR}/partfield/segments/no_dino_animals_12/labels}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/topomatch_semantic_reindexed}

case "${PAIR_ID}" in
  4)
    PAIR_SLUG=bulldog_to_cat
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    SOURCE_NAME=hound_to_bulldog_no_dino
    TARGET_NAME=hound_to_cat_no_dino
    ;;
  5)
    PAIR_SLUG=cat_to_bulldog
    SOURCE=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    SOURCE_NAME=hound_to_cat_no_dino
    TARGET_NAME=hound_to_bulldog_no_dino
    ;;
  6)
    PAIR_SLUG=bulldog_to_dachshund
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    SOURCE_NAME=hound_to_bulldog_no_dino
    TARGET_NAME=hound_to_dachshund
    ;;
  7)
    PAIR_SLUG=dachshund_to_bulldog
    SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    SOURCE_NAME=hound_to_dachshund
    TARGET_NAME=hound_to_bulldog_no_dino
    ;;
  *)
    echo "Unknown PAIR_ID=${PAIR_ID}. Expected 4..7."
    exit 2
    ;;
esac

if [[ -z "${DEFORMED_RUN_DIR}" ]]; then
  DEFORMED_RUN_DIR=$(find "${BASE_DIR}/outputs/topomatch_vcorr" -mindepth 1 -maxdepth 1 -type d \
    -name "${PAIR_SLUG}_*topomatch_vcorr*" 2>/dev/null \
    | sort | tail -n 1)
fi

if [[ -z "${DEFORMED_RUN_DIR}" ]]; then
  DEFORMED_RUN_DIR=$(find "${BASE_DIR}/outputs" -mindepth 2 -maxdepth 2 -type d \
    -path "*/${PAIR_SLUG}_*topomatch_vcorr*" \
    ! -path "*/topomatch_semantic_reindexed/*" \
    | sort | tail -n 1)
fi

if [[ -z "${DEFORMED_RUN_DIR}" || ! -d "${DEFORMED_RUN_DIR}" ]]; then
  echo "Could not find a topomatch deformation directory for ${PAIR_SLUG}."
  exit 1
fi

DEFORMED_MESH="${DEFORMED_RUN_DIR}/mesh_final/mesh.obj"
SOURCE_FEATURES=${SOURCE_FEATURES:-${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_NAME}_0_batch.npy}
TARGET_FEATURES=${TARGET_FEATURES:-${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_NAME}_0_batch.npy}
SOURCE_LABELS=${SOURCE_LABELS:-${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz}
TARGET_LABELS=${TARGET_LABELS:-${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz}
OUT="${OUTPUT_ROOT}/${PAIR_SLUG}_semantic_reindexed_from_$(basename "${DEFORMED_RUN_DIR}")"

for path in "${SOURCE}" "${TARGET}" "${DEFORMED_MESH}" "${SOURCE_FEATURES}" "${TARGET_FEATURES}" "${SOURCE_LABELS}" "${TARGET_LABELS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}"
    exit 1
  fi
done

mkdir -p "${OUT}"

python jobs_with_target_guidance/semantic_reindex_topomatch.py \
  --source-mesh "${SOURCE}" \
  --target-mesh "${TARGET}" \
  --deformed-mesh "${DEFORMED_MESH}" \
  --source-features "${SOURCE_FEATURES}" \
  --target-features "${TARGET_FEATURES}" \
  --source-labels "${SOURCE_LABELS}" \
  --target-labels "${TARGET_LABELS}" \
  --output-dir "${OUT}" \
  --semantic-weight "${SEMANTIC_WEIGHT:-1.0}" \
  --position-weight "${POSITION_WEIGHT:-0.15}" \
  --normal-weight "${NORMAL_WEIGHT:-0.05}" \
  --label-mismatch-penalty "${LABEL_MISMATCH_PENALTY:-0.35}" \
  --topology-prior-weight "${TOPOLOGY_PRIOR_WEIGHT:-0.02}" \
  --topk "${TOPK:-256}"
