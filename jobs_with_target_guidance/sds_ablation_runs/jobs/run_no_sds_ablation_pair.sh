#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: $0 ABLATION TASK_ID EPOCHS OUTPUT_ROOT RUN_TAG"
  echo "  ABLATION: chamfer_only | partfield_chamfer | partfield_chamfer_target_dino"
  echo "  TASK_ID: 0=blueberry_to_g_class, 1=f1_car_to_f1_verstappen, 2=blueberry_to_bugatti, 3=mini_cooper_to_g_class, 4=blueberry_to_santa_fe"
  exit 2
fi

ABLATION="$1"
TASK_ID="$2"
EPOCHS="$3"
OUTPUT_ROOT="$4"
RUN_TAG="$5"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p "${OUTPUT_ROOT}" jobs_with_target_guidance/sds_ablation_runs/logs

source ./activate_meshup_new.sh

BASE_CAR_DIR=./jobs_with_sam3D/meshes/5k_upright_wheels_down
NEW_CAR_DIR=./jobs_with_sam3D/meshes/new_car_meshes_5k_upright_wheels_down
PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/all_car_bucket_visuals_5k
PARTFIELD_LABEL_DIR=./jobs_with_target_guidance/partfield_segments/all_car_bucket_visuals_5k/labels

case "${TASK_ID}" in
  0)
    SOURCE="${BASE_CAR_DIR}/blueberry_5k_upright_wheels_down.ply"
    TARGET="${NEW_CAR_DIR}/g_class_5k_upright_wheels_down.ply"
    SOURCE_LABEL_NAME=blueberry
    TARGET_LABEL_NAME=g_class
    SOURCE_FEATURE_NAME=blueberry_5k_upright_wheels_down
    TARGET_FEATURE_NAME=g_class_5k_upright_wheels_down
    LABEL=blueberry_to_g_class
    PROMPT="an SUV"
    ;;
  1)
    SOURCE="${NEW_CAR_DIR}/f1_car_5k_upright_wheels_down.ply"
    TARGET="${NEW_CAR_DIR}/f1_verstappen_5k_upright_wheels_down.ply"
    SOURCE_LABEL_NAME=f1_car
    TARGET_LABEL_NAME=f1_verstappen
    SOURCE_FEATURE_NAME=f1_car_5k_upright_wheels_down
    TARGET_FEATURE_NAME=f1_verstappen_5k_upright_wheels_down
    LABEL=f1_car_to_f1_verstappen
    PROMPT="an f1 race car"
    ;;
  2)
    SOURCE="${BASE_CAR_DIR}/blueberry_5k_upright_wheels_down.ply"
    TARGET="${BASE_CAR_DIR}/bugatti-centodieci_5k_upright_wheels_down.ply"
    SOURCE_LABEL_NAME=blueberry
    TARGET_LABEL_NAME=bugatti_centodieci
    SOURCE_FEATURE_NAME=blueberry_5k_upright_wheels_down
    TARGET_FEATURE_NAME=bugatti-centodieci_5k_upright_wheels_down
    LABEL=blueberry_to_bugatti
    PROMPT="a sports car"
    ;;
  3)
    SOURCE="${NEW_CAR_DIR}/mini_cooper_5k_upright_wheels_down.ply"
    TARGET="${NEW_CAR_DIR}/g_class_5k_upright_wheels_down.ply"
    SOURCE_LABEL_NAME=mini_cooper
    TARGET_LABEL_NAME=g_class
    SOURCE_FEATURE_NAME=mini_cooper_5k_upright_wheels_down
    TARGET_FEATURE_NAME=g_class_5k_upright_wheels_down
    LABEL=mini_cooper_to_g_class
    PROMPT="an SUV"
    ;;
  4)
    SOURCE="${BASE_CAR_DIR}/blueberry_5k_upright_wheels_down.ply"
    TARGET="${BASE_CAR_DIR}/santa_fe_5k_upright_wheels_down.ply"
    SOURCE_LABEL_NAME=blueberry
    TARGET_LABEL_NAME=santa_fe
    SOURCE_FEATURE_NAME=blueberry_5k_upright_wheels_down
    TARGET_FEATURE_NAME=santa_fe_5k_upright_wheels_down
    LABEL=blueberry_to_santa_fe
    PROMPT="an SUV"
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0..4."
    exit 1
    ;;
esac

SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_FEATURE_NAME}_0_batch.npy
TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_FEATURE_NAME}_0_batch.npy
SOURCE_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${SOURCE_LABEL_NAME}_partfield_labels.npz
TARGET_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${TARGET_LABEL_NAME}_partfield_labels.npz

GLOBAL_CHAMFER_WEIGHT=750
PARTFIELD_CHAMFER_WEIGHT=0
PARTFIELD_ARGS=()
TARGET_GUIDANCE_ARGS=(
  --no-use_target_mesh_guidance
  --target_mesh_weight 0.0
  --target_mesh_render_weight 0.0
)

case "${ABLATION}" in
  chamfer_only)
    PARTFIELD_CHAMFER_WEIGHT=0
    ;;
  partfield_chamfer)
    PARTFIELD_CHAMFER_WEIGHT=8000
    PARTFIELD_ARGS=(
      --partfield_source_features "${SOURCE_PARTFIELD_FEATURES}"
      --partfield_target_features "${TARGET_PARTFIELD_FEATURES}"
      --partfield_source_labels "${SOURCE_PARTFIELD_LABELS}"
      --partfield_target_labels "${TARGET_PARTFIELD_LABELS}"
      --partfield_labels_aligned
      --partfield_label_mode auto
      --partfield_feature_mode auto
      --partfield_n_buckets 12
      --partfield_guidance_mode hard
      --partfield_hard_weight 1.0
      --partfield_soft_weight 0.0
    )
    ;;
  partfield_chamfer_target_dino)
    PARTFIELD_CHAMFER_WEIGHT=8000
    TARGET_GUIDANCE_ARGS=(
      --use_target_mesh_guidance
      --target_mesh_weight 130.0
      --target_mesh_warmup_epochs 15
      --target_mesh_global_weight 1.0
      --target_mesh_spatial_weight 4.0
      --target_mesh_render_weight 8.0
      --target_mesh_n_azimuths 16
      --target_mesh_n_elevations 4
      --target_mesh_online_render
      --target_mesh_online_cache
      --target_mesh_online_cache_max 4096
      --target_mesh_view_rounding_deg 5.0
      --target_mesh_view_rounding_dist 0.1
      --target_mesh_view_rounding_fov 2.0
    )
    PARTFIELD_ARGS=(
      --partfield_source_features "${SOURCE_PARTFIELD_FEATURES}"
      --partfield_target_features "${TARGET_PARTFIELD_FEATURES}"
      --partfield_source_labels "${SOURCE_PARTFIELD_LABELS}"
      --partfield_target_labels "${TARGET_PARTFIELD_LABELS}"
      --partfield_labels_aligned
      --partfield_label_mode auto
      --partfield_feature_mode auto
      --partfield_n_buckets 12
      --partfield_guidance_mode hard
      --partfield_hard_weight 1.0
      --partfield_soft_weight 0.0
    )
    ;;
  *)
    echo "Unknown ABLATION=${ABLATION}. Expected chamfer_only, partfield_chamfer, or partfield_chamfer_target_dino."
    exit 1
    ;;
esac

REQUIRED_PATHS=("${SOURCE}" "${TARGET}")
if [[ "${ABLATION}" == "partfield_chamfer" || "${ABLATION}" == "partfield_chamfer_target_dino" ]]; then
  REQUIRED_PATHS+=(
    "${SOURCE_PARTFIELD_FEATURES}"
    "${TARGET_PARTFIELD_FEATURES}"
    "${SOURCE_PARTFIELD_LABELS}"
    "${TARGET_PARTFIELD_LABELS}"
  )
fi

for path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}"
    exit 1
  fi
done

JOB_STEM=${SLURM_JOB_ID:-manual}
OUT=${OUTPUT_ROOT}/${LABEL}_${ABLATION}_${RUN_TAG}_${EPOCHS}ep_${JOB_STEM}

echo "======================================================"
echo "No-SDS ablation"
echo "ABLATION=${ABLATION}"
echo "TASK_ID=${TASK_ID}"
echo "LABEL=${LABEL}"
echo "EPOCHS=${EPOCHS}"
echo "SOURCE=${SOURCE}"
echo "TARGET=${TARGET}"
echo "PROMPT=${PROMPT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "PARTFIELD_CHAMFER_WEIGHT=${PARTFIELD_CHAMFER_WEIGHT}"
echo "SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS}"
echo "TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS}"
echo "SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES}"
echo "TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES}"
echo "OUT=${OUT}"
echo "START_TIME=$(date)"
echo "======================================================"

CMD=(
  python main.py
  --config ./jobs_with_target_guidance/sds_ablation_runs/configs/car_no_sds_ablation.yml
  --no-use_sds
  --image_weight 0.0
  --loss_schedule independent
  --no-use_dino_loss
  --no-use_cross_attn_loss
  "${TARGET_GUIDANCE_ARGS[@]}"
  --mesh "${SOURCE}"
  --text_prompt "${PROMPT}"
  --target_mesh "${TARGET}"
  --target_mesh_chamfer_weight "${GLOBAL_CHAMFER_WEIGHT}"
  --target_mesh_partfield_chamfer_weight "${PARTFIELD_CHAMFER_WEIGHT}"
  --regularize_jacobians_weight 1800
  --accum_iter 1
  --output_path "${OUT}"
  --epochs "${EPOCHS}"
  "${PARTFIELD_ARGS[@]}"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  python jobs_with_target_guidance/evaluate_target_pipeline.py \
    --output-dir "${OUT}" \
    --samples "${EVAL_SAMPLES:-3000}" \
    --part-samples "${EVAL_PART_SAMPLES:-750}"
fi

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
