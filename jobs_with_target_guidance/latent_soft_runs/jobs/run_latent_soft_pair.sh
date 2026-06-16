#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: $0 TASK_ID EPOCHS OUTPUT_ROOT RUN_TAG"
  exit 2
fi

TASK_ID="$1"
EPOCHS="$2"
OUTPUT_ROOT="$3"
RUN_TAG="$4"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p "${OUTPUT_ROOT}" jobs_with_target_guidance/latent_soft_runs/logs

source ./activate_meshup_new.sh

PARTFIELD_CHAMFER_WEIGHT=8000
GLOBAL_CHAMFER_WEIGHT=750
TARGET_WEIGHT=130
TARGET_RENDER_WEIGHT=8
REG_WEIGHT=1800
SOFT_POINTS=1024
SOFT_SEMANTIC_WEIGHT=1.0
SOFT_TEMPERATURE=0.05
N_BUCKETS=12

case "${TASK_ID}" in
  0)
    GROUP=dog
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    SOURCE_NAME=hound_to_bulldog
    TARGET_NAME=hound_to_dachshund
    LABEL=bulldog_to_dachshund
    PROMPT="a dog"
    ;;
  1)
    GROUP=dog
    SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_bulldog_exp/mesh_final/mesh.obj
    SOURCE_NAME=hound_to_dachshund
    TARGET_NAME=hound_to_bulldog
    LABEL=dachshund_to_bulldog
    PROMPT="a dog"
    ;;
  2)
    GROUP=car
    SOURCE=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/f1_car_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/f1_verstappen_5k_upright_wheels_down.ply
    SOURCE_NAME=f1_car
    TARGET_NAME=f1_verstappen
    SOURCE_FEATURE_NAME=f1_car_5k_upright_wheels_down
    TARGET_FEATURE_NAME=f1_verstappen_5k_upright_wheels_down
    LABEL=f1_car_to_f1_verstappen
    PROMPT="an f1 race car"
    ;;
  3)
    GROUP=car
    SOURCE=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/f1_verstappen_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/f1_car_5k_upright_wheels_down.ply
    SOURCE_NAME=f1_verstappen
    TARGET_NAME=f1_car
    SOURCE_FEATURE_NAME=f1_verstappen_5k_upright_wheels_down
    TARGET_FEATURE_NAME=f1_car_5k_upright_wheels_down
    LABEL=f1_verstappen_to_f1_car
    PROMPT="an f1 race car"
    ;;
  4)
    GROUP=car
    SOURCE=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/mini_cooper_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/g_class_5k_upright_wheels_down.ply
    SOURCE_NAME=mini_cooper
    TARGET_NAME=g_class
    SOURCE_FEATURE_NAME=mini_cooper_5k_upright_wheels_down
    TARGET_FEATURE_NAME=g_class_5k_upright_wheels_down
    LABEL=mini_cooper_to_g_class
    PROMPT="an SUV"
    ;;
  5)
    GROUP=car
    SOURCE=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/red_truck1_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/red_truck2_5k_upright_wheels_down.ply
    SOURCE_NAME=red_truck1
    TARGET_NAME=red_truck2
    SOURCE_FEATURE_NAME=red_truck1_5k_upright_wheels_down
    TARGET_FEATURE_NAME=red_truck2_5k_upright_wheels_down
    LABEL=red_truck1_to_red_truck2
    PROMPT="a truck car"
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0..5."
    exit 1
    ;;
esac

if [[ "${GROUP}" == "dog" ]]; then
  PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/test_meshes
  PARTFIELD_LABEL_DIR=./jobs_with_target_guidance/partfield_segments/dachshund_bulldog_12/labels
  SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_NAME}_0_batch.npy
  TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_NAME}_0_batch.npy
else
  PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/new_car_target_runs_5k
  PARTFIELD_LABEL_DIR=./jobs_with_target_guidance/partfield_segments/new_car_target_runs_5k/labels
  SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_FEATURE_NAME}_0_batch.npy
  TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_FEATURE_NAME}_0_batch.npy
fi

SOURCE_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz
TARGET_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz

for path in \
  "${SOURCE}" \
  "${TARGET}" \
  "${SOURCE_PARTFIELD_LABELS}" \
  "${TARGET_PARTFIELD_LABELS}" \
  "${SOURCE_PARTFIELD_FEATURES}" \
  "${TARGET_PARTFIELD_FEATURES}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}"
    exit 1
  fi
done

JOB_STEM=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
OUT=${OUTPUT_ROOT}/${LABEL}_${RUN_TAG}_latent_soft_${EPOCHS}ep_${JOB_STEM}_${TASK_ID}

echo "======================================================"
echo "PartField latent-soft target guidance"
echo "TASK_ID=${TASK_ID}"
echo "GROUP=${GROUP}"
echo "LABEL=${LABEL}"
echo "EPOCHS=${EPOCHS}"
echo "SOURCE=${SOURCE}"
echo "TARGET=${TARGET}"
echo "PROMPT=${PROMPT}"
echo "SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS}"
echo "TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS}"
echo "SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES}"
echo "TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES}"
echo "OUT=${OUT}"
echo "SOFT_POINTS=${SOFT_POINTS}"
echo "SOFT_SEMANTIC_WEIGHT=${SOFT_SEMANTIC_WEIGHT}"
echo "SOFT_TEMPERATURE=${SOFT_TEMPERATURE}"
echo "START_TIME=$(date)"
echo "======================================================"

python main.py \
  --config ./jobs_with_target_guidance/configs/car_partfield_chamfer.yml \
  --mesh "${SOURCE}" \
  --text_prompt "${PROMPT}" \
  --target_mesh "${TARGET}" \
  --target_mesh_weight "${TARGET_WEIGHT}" \
  --target_mesh_render_weight "${TARGET_RENDER_WEIGHT}" \
  --target_mesh_chamfer_weight "${GLOBAL_CHAMFER_WEIGHT}" \
  --target_mesh_partfield_chamfer_weight "${PARTFIELD_CHAMFER_WEIGHT}" \
  --partfield_source_labels "${SOURCE_PARTFIELD_LABELS}" \
  --partfield_target_labels "${TARGET_PARTFIELD_LABELS}" \
  --partfield_source_features "${SOURCE_PARTFIELD_FEATURES}" \
  --partfield_target_features "${TARGET_PARTFIELD_FEATURES}" \
  --partfield_labels_aligned \
  --partfield_label_mode auto \
  --partfield_feature_mode auto \
  --partfield_n_buckets "${N_BUCKETS}" \
  --partfield_guidance_mode soft \
  --partfield_hard_weight 0.0 \
  --partfield_soft_weight 1.0 \
  --partfield_soft_points "${SOFT_POINTS}" \
  --partfield_soft_semantic_weight "${SOFT_SEMANTIC_WEIGHT}" \
  --partfield_soft_temperature "${SOFT_TEMPERATURE}" \
  --partfield_soft_match_space latent \
  --regularize_jacobians_weight "${REG_WEIGHT}" \
  --output_path "${OUT}" \
  --epochs "${EPOCHS}"

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  python jobs_with_target_guidance/evaluate_target_pipeline.py \
    --output-dir "${OUT}" \
    --samples "${EVAL_SAMPLES:-3000}" \
    --part-samples "${EVAL_PART_SAMPLES:-750}"
fi

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
