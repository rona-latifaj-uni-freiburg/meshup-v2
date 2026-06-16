#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: $0 PAIR_ID VARIANT_ID EPOCHS OUTPUT_ROOT RUN_TAG"
  exit 2
fi

PAIR_ID="$1"
VARIANT_ID="$2"
EPOCHS="$3"
OUTPUT_ROOT="$4"
RUN_TAG="$5"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p "${OUTPUT_ROOT}" jobs_with_target_guidance/artur_soft_runs/logs

source ./activate_meshup_new.sh

PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/all_car_bucket_visuals_5k
PARTFIELD_LABEL_DIR=./jobs_with_target_guidance/partfield_segments/all_car_bucket_visuals_5k/labels
N_BUCKETS=12

case "${PAIR_ID}" in
  0)
    PAIR=blueberry_to_santa_fe
    SOURCE=./jobs_with_sam3D/meshes/5k_upright_wheels_down/blueberry_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/5k_upright_wheels_down/santa_fe_5k_upright_wheels_down.ply
    SOURCE_NAME=blueberry
    TARGET_NAME=santa_fe
    SOURCE_FEATURE_NAME=blueberry_5k_upright_wheels_down
    TARGET_FEATURE_NAME=santa_fe_5k_upright_wheels_down
    PROMPT="a car"
    ;;
  1)
    PAIR=f1_car_to_f1_verstappen
    SOURCE=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/f1_car_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/f1_verstappen_5k_upright_wheels_down.ply
    SOURCE_NAME=f1_car
    TARGET_NAME=f1_verstappen
    SOURCE_FEATURE_NAME=f1_car_5k_upright_wheels_down
    TARGET_FEATURE_NAME=f1_verstappen_5k_upright_wheels_down
    PROMPT="an f1 race car"
    ;;
  2)
    PAIR=mini_cooper_to_g_class
    SOURCE=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/mini_cooper_5k_upright_wheels_down.ply
    TARGET=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down/g_class_5k_upright_wheels_down.ply
    SOURCE_NAME=mini_cooper
    TARGET_NAME=g_class
    SOURCE_FEATURE_NAME=mini_cooper_5k_upright_wheels_down
    TARGET_FEATURE_NAME=g_class_5k_upright_wheels_down
    PROMPT="a car"
    ;;
  *)
    echo "Unknown PAIR_ID=${PAIR_ID}. Expected 0, 1, or 2."
    exit 1
    ;;
esac

SOURCE_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz
TARGET_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz
SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_FEATURE_NAME}_0_batch.npy
TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_FEATURE_NAME}_0_batch.npy

GLOBAL_CHAMFER_WEIGHT=0.0
PARTFIELD_CHAMFER_WEIGHT=0.0
PARTFIELD_GUIDANCE_MODE=hard
PARTFIELD_HARD_WEIGHT=0.0
PARTFIELD_SOFT_WEIGHT=0.0
ARTUR_SOFT_MATCH_SPACE=hybrid
ARTUR_SOFT_POINTS=1024
ARTUR_SOFT_GEOMETRY_SIGMA=0.5
ARTUR_SOFT_SEMANTIC_WEIGHT=1.0
ARTUR_SOFT_TEMPERATURE=0.1

case "${VARIANT_ID}" in
  0)
    VARIANT=global_chamfer_only
    GLOBAL_CHAMFER_WEIGHT=750.0
    ;;
  1)
    VARIANT=hard_partfield_chamfer_only
    PARTFIELD_CHAMFER_WEIGHT=8000.0
    PARTFIELD_GUIDANCE_MODE=hard
    PARTFIELD_HARD_WEIGHT=1.0
    PARTFIELD_SOFT_WEIGHT=0.0
    ;;
  2)
    VARIANT=artur_soft_partfield_chamfer_only
    PARTFIELD_CHAMFER_WEIGHT=8000.0
    PARTFIELD_GUIDANCE_MODE=soft
    PARTFIELD_HARD_WEIGHT=0.0
    PARTFIELD_SOFT_WEIGHT=1.0
    ;;
  *)
    echo "Unknown VARIANT_ID=${VARIANT_ID}. Expected 0, 1, or 2."
    exit 1
    ;;
esac

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

JOB_STEM=${SLURM_JOB_ID:-manual}
OUT=${OUTPUT_ROOT}/${PAIR}_${RUN_TAG}_${VARIANT}_${EPOCHS}ep_${JOB_STEM}

echo "======================================================"
echo "Artur soft car Chamfer ablation"
echo "PAIR_ID=${PAIR_ID}"
echo "PAIR=${PAIR}"
echo "VARIANT_ID=${VARIANT_ID}"
echo "VARIANT=${VARIANT}"
echo "EPOCHS=${EPOCHS}"
echo "SOURCE=${SOURCE}"
echo "TARGET=${TARGET}"
echo "PROMPT=${PROMPT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "PARTFIELD_CHAMFER_WEIGHT=${PARTFIELD_CHAMFER_WEIGHT}"
echo "PARTFIELD_GUIDANCE_MODE=${PARTFIELD_GUIDANCE_MODE}"
echo "ARTUR_SOFT_MATCH_SPACE=${ARTUR_SOFT_MATCH_SPACE}"
echo "ARTUR_SOFT_GEOMETRY_SIGMA=${ARTUR_SOFT_GEOMETRY_SIGMA}"
echo "ARTUR_SOFT_SEMANTIC_WEIGHT=${ARTUR_SOFT_SEMANTIC_WEIGHT}"
echo "ARTUR_SOFT_TEMPERATURE=${ARTUR_SOFT_TEMPERATURE}"
echo "OUT=${OUT}"
echo "START_TIME=$(date)"
echo "======================================================"

python main.py \
  --config ./jobs_with_target_guidance/configs/car_partfield_chamfer.yml \
  --mesh "${SOURCE}" \
  --text_prompt "${PROMPT}" \
  --target_mesh "${TARGET}" \
  --no-use_sds \
  --image_weight 0.0 \
  --image_weight_start_factor 0.0 \
  --image_weight_ramp_epochs 0 \
  --loss_schedule independent \
  --regularize_jacobians_weight 0.0 \
  --no-use_dino_loss \
  --no-use_cross_attn_loss \
  --no-use_target_mesh_guidance \
  --target_mesh_weight 0.0 \
  --target_mesh_render_weight 0.0 \
  --target_mesh_chamfer_weight "${GLOBAL_CHAMFER_WEIGHT}" \
  --target_mesh_chamfer_warmup_epochs 0 \
  --target_mesh_chamfer_points 3072 \
  --target_mesh_partfield_chamfer_weight "${PARTFIELD_CHAMFER_WEIGHT}" \
  --target_mesh_partfield_chamfer_warmup_epochs 0 \
  --target_mesh_partfield_chamfer_points 512 \
  --target_mesh_partfield_chamfer_global_weight 0.0 \
  --partfield_source_labels "${SOURCE_PARTFIELD_LABELS}" \
  --partfield_target_labels "${TARGET_PARTFIELD_LABELS}" \
  --partfield_source_features "${SOURCE_PARTFIELD_FEATURES}" \
  --partfield_target_features "${TARGET_PARTFIELD_FEATURES}" \
  --partfield_labels_aligned \
  --partfield_label_mode auto \
  --partfield_feature_mode auto \
  --partfield_n_buckets "${N_BUCKETS}" \
  --partfield_guidance_mode "${PARTFIELD_GUIDANCE_MODE}" \
  --partfield_hard_weight "${PARTFIELD_HARD_WEIGHT}" \
  --partfield_soft_weight "${PARTFIELD_SOFT_WEIGHT}" \
  --partfield_soft_points "${ARTUR_SOFT_POINTS}" \
  --partfield_soft_match_space "${ARTUR_SOFT_MATCH_SPACE}" \
  --partfield_soft_geometry_sigma "${ARTUR_SOFT_GEOMETRY_SIGMA}" \
  --partfield_soft_semantic_weight "${ARTUR_SOFT_SEMANTIC_WEIGHT}" \
  --partfield_soft_temperature "${ARTUR_SOFT_TEMPERATURE}" \
  --log_interval_im 250 \
  --save_renders_interval 250 \
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
