#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: $0 VARIANT_ID EPOCHS OUTPUT_ROOT RUN_TAG"
  exit 2
fi

VARIANT_ID="$1"
EPOCHS="$2"
OUTPUT_ROOT="$3"
RUN_TAG="$4"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p "${OUTPUT_ROOT}" jobs_with_target_guidance/artur_soft_runs/logs

source ./activate_meshup_new.sh

SOURCE=${SOURCE:-./experiments/dog_morphs/outputs/hound_to_bulldog_exp/mesh_final/mesh.obj}
TARGET=${TARGET:-./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj}
SOURCE_NAME=${SOURCE_NAME:-hound_to_bulldog}
TARGET_NAME=${TARGET_NAME:-hound_to_dachshund}
PAIR_SLUG=${PAIR_SLUG:-bulldog_to_dachshund}
PROMPT=${PROMPT:-"a dog"}

PARTFIELD_FEATURE_DIR=${PARTFIELD_FEATURE_DIR:-./jobs_with_target_guidance/partfield_features/test_meshes}
SOURCE_PARTFIELD_FEATURE_DIR=${SOURCE_PARTFIELD_FEATURE_DIR:-${PARTFIELD_FEATURE_DIR}}
TARGET_PARTFIELD_FEATURE_DIR=${TARGET_PARTFIELD_FEATURE_DIR:-${PARTFIELD_FEATURE_DIR}}
PARTFIELD_LABEL_DIR=${PARTFIELD_LABEL_DIR:-./jobs_with_target_guidance/partfield_segments/dachshund_bulldog_12/labels}
SOURCE_PARTFIELD_LABEL_DIR=${SOURCE_PARTFIELD_LABEL_DIR:-${PARTFIELD_LABEL_DIR}}
TARGET_PARTFIELD_LABEL_DIR=${TARGET_PARTFIELD_LABEL_DIR:-${PARTFIELD_LABEL_DIR}}
SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS:-${SOURCE_PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz}
TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS:-${TARGET_PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz}
SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES:-${SOURCE_PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_NAME}_0_batch.npy}
TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES:-${TARGET_PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_NAME}_0_batch.npy}

N_BUCKETS=12
DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-jacobian}
JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-0.0}
JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
JACOBIAN_OUTLIER_POWER=${JACOBIAN_OUTLIER_POWER:-4.0}
DEFORMATION_GRAD_CLIP_NORM=${DEFORMATION_GRAD_CLIP_NORM:-0.0}
EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
EDGE_STRETCH_THRESHOLD=${EDGE_STRETCH_THRESHOLD:-1.5}
EDGE_STRETCH_MAX_WEIGHT=${EDGE_STRETCH_MAX_WEIGHT:-1.0}
EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-0.0}
EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.25}
EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-1.0}
TARGET_CHAMFER_WARMUP_EPOCHS=${TARGET_CHAMFER_WARMUP_EPOCHS:-0}
PARTFIELD_CHAMFER_WARMUP_EPOCHS=${PARTFIELD_CHAMFER_WARMUP_EPOCHS:-0}
LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-250}
SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-${LOG_INTERVAL_IM}}
GLOBAL_CHAMFER_WEIGHT=0.0
PARTFIELD_CHAMFER_WEIGHT=0.0
PARTFIELD_GUIDANCE_MODE_OVERRIDE=${PARTFIELD_GUIDANCE_MODE:-}
PARTFIELD_HARD_WEIGHT_OVERRIDE=${PARTFIELD_HARD_WEIGHT:-}
PARTFIELD_SOFT_WEIGHT_OVERRIDE=${PARTFIELD_SOFT_WEIGHT:-}
PARTFIELD_GUIDANCE_MODE=hard
PARTFIELD_HARD_WEIGHT=0.0
PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT:-1.0}
PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT:-1.0}
PARTFIELD_SOFT_WEIGHT=0.0
ARTUR_SOFT_MATCH_SPACE=${ARTUR_SOFT_MATCH_SPACE:-hybrid}
ARTUR_SOFT_POINTS=${ARTUR_SOFT_POINTS:-1024}
ARTUR_SOFT_GEOMETRY_SIGMA=${ARTUR_SOFT_GEOMETRY_SIGMA:-0.5}
ARTUR_SOFT_SEMANTIC_WEIGHT=${ARTUR_SOFT_SEMANTIC_WEIGHT:-1.0}
ARTUR_SOFT_TEMPERATURE=${ARTUR_SOFT_TEMPERATURE:-0.1}
PARTFIELD_BALANCED_SINKHORN_ITERS=${PARTFIELD_BALANCED_SINKHORN_ITERS:-30}
PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
PARTFIELD_CONTAINMENT_MARGIN=${PARTFIELD_CONTAINMENT_MARGIN:-0.02}
PARTFIELD_CONTAINMENT_MAX_WEIGHT=${PARTFIELD_CONTAINMENT_MAX_WEIGHT:-1.0}

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

if [[ -n "${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-}" ]]; then
  GLOBAL_CHAMFER_WEIGHT="${GLOBAL_CHAMFER_WEIGHT_OVERRIDE}"
fi
if [[ -n "${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-}" ]]; then
  PARTFIELD_CHAMFER_WEIGHT="${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE}"
fi
if [[ -n "${PARTFIELD_GUIDANCE_MODE_OVERRIDE}" ]]; then
  PARTFIELD_GUIDANCE_MODE="${PARTFIELD_GUIDANCE_MODE_OVERRIDE}"
fi
if [[ -n "${PARTFIELD_HARD_WEIGHT_OVERRIDE}" ]]; then
  PARTFIELD_HARD_WEIGHT="${PARTFIELD_HARD_WEIGHT_OVERRIDE}"
fi
if [[ -n "${PARTFIELD_SOFT_WEIGHT_OVERRIDE}" ]]; then
  PARTFIELD_SOFT_WEIGHT="${PARTFIELD_SOFT_WEIGHT_OVERRIDE}"
fi
if [[ -n "${VARIANT_SUFFIX:-}" ]]; then
  VARIANT="${VARIANT}_${VARIANT_SUFFIX}"
fi

DINO_LOSS_ARGS=(--no-use_dino_loss)
if [[ "${ENABLE_SOURCE_DINO_LOSS:-0}" == "1" ]]; then
  DINO_LOSS_ARGS=(
    --use_dino_loss
    --dino_weight "${SOURCE_DINO_WEIGHT:-0.08}"
    --dino_warmup_epochs "${SOURCE_DINO_WARMUP_EPOCHS:-100}"
  )
fi

TARGET_DINO_ARGS=(
  --no-use_target_mesh_guidance
  --target_mesh_weight 0.0
  --target_mesh_render_weight 0.0
)
if [[ "${ENABLE_TARGET_DINO_GUIDANCE:-0}" == "1" ]]; then
  TARGET_DINO_ARGS=(
    --use_target_mesh_guidance
    --target_mesh_weight "${TARGET_DINO_WEIGHT:-130.0}"
    --target_mesh_render_weight "${TARGET_DINO_RENDER_WEIGHT:-8.0}"
    --target_mesh_global_weight "${TARGET_DINO_GLOBAL_WEIGHT:-1.0}"
    --target_mesh_spatial_weight "${TARGET_DINO_SPATIAL_WEIGHT:-4.0}"
    --target_mesh_warmup_epochs "${TARGET_DINO_WARMUP_EPOCHS:-15}"
    --target_mesh_n_azimuths "${TARGET_DINO_N_AZIMUTHS:-16}"
    --target_mesh_n_elevations "${TARGET_DINO_N_ELEVATIONS:-4}"
    --target_mesh_online_render
    --target_mesh_online_cache
    --target_mesh_online_cache_max "${TARGET_DINO_ONLINE_CACHE_MAX:-4096}"
    --target_mesh_view_rounding_deg "${TARGET_DINO_VIEW_ROUNDING_DEG:-5.0}"
    --target_mesh_view_rounding_dist "${TARGET_DINO_VIEW_ROUNDING_DIST:-0.1}"
    --target_mesh_view_rounding_fov "${TARGET_DINO_VIEW_ROUNDING_FOV:-2.0}"
  )
fi

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
OUT=${OUTPUT_ROOT}/${PAIR_SLUG}_${RUN_TAG}_${VARIANT}_${EPOCHS}ep_${JOB_STEM}

echo "======================================================"
echo "Artur soft Chamfer ablation"
echo "VARIANT_ID=${VARIANT_ID}"
echo "VARIANT=${VARIANT}"
echo "EPOCHS=${EPOCHS}"
echo "SOURCE=${SOURCE}"
echo "TARGET=${TARGET}"
echo "PROMPT=${PROMPT}"
echo "GLOBAL_CHAMFER_WEIGHT=${GLOBAL_CHAMFER_WEIGHT}"
echo "PARTFIELD_CHAMFER_WEIGHT=${PARTFIELD_CHAMFER_WEIGHT}"
echo "DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION}"
echo "JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT}"
echo "JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT}"
echo "JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT}"
echo "JACOBIAN_OUTLIER_POWER=${JACOBIAN_OUTLIER_POWER}"
echo "DEFORMATION_GRAD_CLIP_NORM=${DEFORMATION_GRAD_CLIP_NORM}"
echo "EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT}"
echo "EDGE_STRETCH_THRESHOLD=${EDGE_STRETCH_THRESHOLD}"
echo "EDGE_STRETCH_MAX_WEIGHT=${EDGE_STRETCH_MAX_WEIGHT}"
echo "EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT}"
echo "EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD}"
echo "EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT}"
echo "TARGET_CHAMFER_WARMUP_EPOCHS=${TARGET_CHAMFER_WARMUP_EPOCHS}"
echo "PARTFIELD_CHAMFER_WARMUP_EPOCHS=${PARTFIELD_CHAMFER_WARMUP_EPOCHS}"
echo "ENABLE_SOURCE_DINO_LOSS=${ENABLE_SOURCE_DINO_LOSS:-0}"
echo "ENABLE_TARGET_DINO_GUIDANCE=${ENABLE_TARGET_DINO_GUIDANCE:-0}"
echo "PARTFIELD_GUIDANCE_MODE=${PARTFIELD_GUIDANCE_MODE}"
echo "PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT}"
echo "PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT}"
echo "ARTUR_SOFT_MATCH_SPACE=${ARTUR_SOFT_MATCH_SPACE}"
echo "ARTUR_SOFT_GEOMETRY_SIGMA=${ARTUR_SOFT_GEOMETRY_SIGMA}"
echo "ARTUR_SOFT_SEMANTIC_WEIGHT=${ARTUR_SOFT_SEMANTIC_WEIGHT}"
echo "ARTUR_SOFT_TEMPERATURE=${ARTUR_SOFT_TEMPERATURE}"
echo "PARTFIELD_BALANCED_SINKHORN_ITERS=${PARTFIELD_BALANCED_SINKHORN_ITERS}"
echo "PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT}"
echo "PARTFIELD_CONTAINMENT_MARGIN=${PARTFIELD_CONTAINMENT_MARGIN}"
echo "PARTFIELD_CONTAINMENT_MAX_WEIGHT=${PARTFIELD_CONTAINMENT_MAX_WEIGHT}"
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
  --deformation_parameterization "${DEFORMATION_PARAMETERIZATION}" \
  --regularize_jacobians_weight "${JACOBIAN_REG_WEIGHT}" \
  --jacobian_neighbor_smooth_weight "${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT}" \
  --jacobian_outlier_weight "${JACOBIAN_OUTLIER_WEIGHT}" \
  --jacobian_outlier_power "${JACOBIAN_OUTLIER_POWER}" \
  --deformation_grad_clip_norm "${DEFORMATION_GRAD_CLIP_NORM}" \
  --edge_stretch_weight "${EDGE_STRETCH_WEIGHT}" \
  --edge_stretch_threshold "${EDGE_STRETCH_THRESHOLD}" \
  --edge_stretch_max_weight "${EDGE_STRETCH_MAX_WEIGHT}" \
  --edge_displacement_jump_weight "${EDGE_DISPLACEMENT_JUMP_WEIGHT}" \
  --edge_displacement_jump_threshold "${EDGE_DISPLACEMENT_JUMP_THRESHOLD}" \
  --edge_displacement_jump_max_weight "${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT}" \
  "${DINO_LOSS_ARGS[@]}" \
  --no-use_cross_attn_loss \
  "${TARGET_DINO_ARGS[@]}" \
  --target_mesh_chamfer_weight "${GLOBAL_CHAMFER_WEIGHT}" \
  --target_mesh_chamfer_warmup_epochs "${TARGET_CHAMFER_WARMUP_EPOCHS}" \
  --target_mesh_chamfer_points 3072 \
  --target_mesh_partfield_chamfer_weight "${PARTFIELD_CHAMFER_WEIGHT}" \
  --target_mesh_partfield_chamfer_warmup_epochs "${PARTFIELD_CHAMFER_WARMUP_EPOCHS}" \
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
  --partfield_source_to_target_weight "${PARTFIELD_SOURCE_TO_TARGET_WEIGHT}" \
  --partfield_target_to_source_weight "${PARTFIELD_TARGET_TO_SOURCE_WEIGHT}" \
  --partfield_soft_weight "${PARTFIELD_SOFT_WEIGHT}" \
  --partfield_soft_points "${ARTUR_SOFT_POINTS}" \
  --partfield_soft_match_space "${ARTUR_SOFT_MATCH_SPACE}" \
  --partfield_soft_geometry_sigma "${ARTUR_SOFT_GEOMETRY_SIGMA}" \
  --partfield_soft_semantic_weight "${ARTUR_SOFT_SEMANTIC_WEIGHT}" \
  --partfield_soft_temperature "${ARTUR_SOFT_TEMPERATURE}" \
  --partfield_balanced_sinkhorn_iters "${PARTFIELD_BALANCED_SINKHORN_ITERS}" \
  --partfield_containment_weight "${PARTFIELD_CONTAINMENT_WEIGHT}" \
  --partfield_containment_margin "${PARTFIELD_CONTAINMENT_MARGIN}" \
  --partfield_containment_max_weight "${PARTFIELD_CONTAINMENT_MAX_WEIGHT}" \
  --log_interval_im "${LOG_INTERVAL_IM}" \
  ${EXTRA_LOG_EPOCHS:+--extra_log_epochs ${EXTRA_LOG_EPOCHS}} \
  --save_renders_interval "${SAVE_RENDERS_INTERVAL}" \
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
