#!/bin/bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: $0 PAIR_ID EPOCHS OUTPUT_ROOT RUN_TAG MODE"
  echo "MODE: global_chamfer | pf_chamfer | pf_chamfer_jneighbor"
  exit 2
fi

PAIR_ID="$1"
EPOCHS="$2"
OUTPUT_ROOT="$3"
RUN_TAG="$4"
MODE="$5"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p "${OUTPUT_ROOT}" jobs_with_target_guidance/artur_soft_runs/logs

source ./activate_meshup_new.sh

PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/no_dino_animals
PARTFIELD_LABEL_DIR=./jobs_with_target_guidance/partfield_segments/no_dino_animals_12/labels
N_BUCKETS=12

case "${PAIR_ID}" in
  0)
    PAIR=bulldog_to_horse
    SOURCE_NAME=hound_to_bulldog_no_dino
    TARGET_NAME=hound_to_horse_no_dino
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_horse_no_dino_exp/mesh_final/mesh.obj
    PROMPT="a horse"
    ;;
  1)
    PAIR=bulldog_to_cat
    SOURCE_NAME=hound_to_bulldog_no_dino
    TARGET_NAME=hound_to_cat_no_dino
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj
    PROMPT="a cat"
    ;;
  2)
    PAIR=bulldog_to_golden_retriever
    SOURCE_NAME=hound_to_bulldog_no_dino
    TARGET_NAME=hound_to_golden_retriever_no_dino
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_golden_retriever_no_dino_exp/mesh_final/mesh.obj
    PROMPT="a golden retriever dog"
    ;;
  3)
    PAIR=bulldog_to_bear
    SOURCE_NAME=hound_to_bulldog_no_dino
    TARGET_NAME=hound_to_bear_no_dino
    SOURCE=./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_bear_no_dino_exp/mesh_final/mesh.obj
    PROMPT="a bear"
    ;;
  4)
    PAIR=dachshund_to_golden_retriever
    SOURCE_NAME=hound_to_dachshund
    TARGET_NAME=hound_to_golden_retriever_no_dino
    SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
    TARGET=./experiments/dog_morphs/outputs/hound_to_golden_retriever_no_dino_exp/mesh_final/mesh.obj
    PROMPT="a golden retriever dog"
    ;;
  *)
    echo "Unknown PAIR_ID=${PAIR_ID}. Expected 0, 1, 2, 3, or 4."
    exit 1
    ;;
esac

SOURCE_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz
TARGET_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz
SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_NAME}_0_batch.npy
TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_NAME}_0_batch.npy

GLOBAL_CHAMFER_WEIGHT=0.0
PARTFIELD_CHAMFER_WEIGHT=0.0
PARTFIELD_GUIDANCE_MODE=hard
PARTFIELD_HARD_WEIGHT=0.0
PARTFIELD_SOFT_WEIGHT=0.0
DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-jacobian}
JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-0.0}
JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
JACOBIAN_OUTLIER_POWER=${JACOBIAN_OUTLIER_POWER:-4.0}
DEFORMATION_GRAD_CLIP_NORM=${DEFORMATION_GRAD_CLIP_NORM:-0.0}
TARGET_CHAMFER_WARMUP_EPOCHS=${TARGET_CHAMFER_WARMUP_EPOCHS:-0}
PARTFIELD_CHAMFER_WARMUP_EPOCHS=${PARTFIELD_CHAMFER_WARMUP_EPOCHS:-0}
LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-250}
SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-250}
EXTRA_LOG_EPOCHS=${EXTRA_LOG_EPOCHS:-"1 10 20 30 40 50 60 70 80 90 100"}

case "${MODE}" in
  global_chamfer)
    VARIANT=global_chamfer_only
    GLOBAL_CHAMFER_WEIGHT=750.0
    PARTFIELD_CHAMFER_WEIGHT=0.0
    PARTFIELD_HARD_WEIGHT=0.0
    ;;
  pf_chamfer)
    VARIANT=hard_partfield_chamfer_only
    GLOBAL_CHAMFER_WEIGHT=0.0
    PARTFIELD_CHAMFER_WEIGHT=8000.0
    PARTFIELD_HARD_WEIGHT=1.0
    ;;
  pf_chamfer_jneighbor)
    VARIANT=hard_partfield_chamfer_only_jneighbor${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT}
    GLOBAL_CHAMFER_WEIGHT=0.0
    PARTFIELD_CHAMFER_WEIGHT=8000.0
    PARTFIELD_HARD_WEIGHT=1.0
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected global_chamfer, pf_chamfer, or pf_chamfer_jneighbor."
    exit 2
    ;;
esac

if [[ -n "${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-}" ]]; then
  GLOBAL_CHAMFER_WEIGHT="${GLOBAL_CHAMFER_WEIGHT_OVERRIDE}"
fi
if [[ -n "${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-}" ]]; then
  PARTFIELD_CHAMFER_WEIGHT="${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE}"
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
OUT=${OUTPUT_ROOT}/${PAIR}_${RUN_TAG}_${VARIANT}_${EPOCHS}ep_${JOB_STEM}

echo "======================================================"
echo "No-DINO animal PartField+Chamfer run"
echo "PAIR_ID=${PAIR_ID}"
echo "PAIR=${PAIR}"
echo "MODE=${MODE}"
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
echo "LOG_INTERVAL_IM=${LOG_INTERVAL_IM}"
echo "SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL}"
echo "EXTRA_LOG_EPOCHS=${EXTRA_LOG_EPOCHS}"
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
  --no-use_dino_loss \
  --no-use_cross_attn_loss \
  --no-use_target_mesh_guidance \
  --target_mesh_weight 0.0 \
  --target_mesh_render_weight 0.0 \
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
  --partfield_soft_weight "${PARTFIELD_SOFT_WEIGHT}" \
  --log_interval_im "${LOG_INTERVAL_IM}" \
  ${EXTRA_LOG_EPOCHS:+--extra_log_epochs ${EXTRA_LOG_EPOCHS}} \
  --save_renders_interval "${SAVE_RENDERS_INTERVAL}" \
  --output_path "${OUT}" \
  --epochs "${EPOCHS}"

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
