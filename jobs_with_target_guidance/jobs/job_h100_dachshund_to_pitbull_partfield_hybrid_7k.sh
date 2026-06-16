#!/bin/bash
#SBATCH --job-name=h100_dach_pfhy7k
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=jobs_with_target_guidance/logs/h100_dach_pfhy7k_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/h100_dach_pfhy7k_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,COMPLETE
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/logs jobs_with_target_guidance/outputs jobs_with_target_guidance/slurm_logs

source ./activate_meshup_new.sh

SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
TARGET=./experiments/dog_morphs/outputs/hound_to_bulldog_exp/mesh_final/mesh.obj
SOURCE_NAME=hound_to_dachshund
TARGET_NAME=hound_to_bulldog
LABEL=dachshund_to_pitbull_hybrid_7k
PROMPT="a pitbull dog with short folded ears and a strong head"

# No literal pitbull mesh is present; hound_to_bulldog is the available dog target proxy.
PARTFIELD_LABEL_DIR=./jobs_with_target_guidance/partfield_segments/dachshund_bulldog_12/labels
PARTFIELD_FEATURE_DIR=./jobs_with_target_guidance/partfield_features/test_meshes
SOURCE_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${SOURCE_NAME}_partfield_labels.npz
TARGET_PARTFIELD_LABELS=${PARTFIELD_LABEL_DIR}/${TARGET_NAME}_partfield_labels.npz
SOURCE_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${SOURCE_NAME}_0_batch.npy
TARGET_PARTFIELD_FEATURES=${PARTFIELD_FEATURE_DIR}/part_feat_${TARGET_NAME}_0_batch.npy

for path in "${SOURCE_PARTFIELD_LABELS}" "${TARGET_PARTFIELD_LABELS}" "${SOURCE_PARTFIELD_FEATURES}" "${TARGET_PARTFIELD_FEATURES}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required PartField input: ${path}"
    exit 1
  fi
done

PARTFIELD_CHAMFER_WEIGHT=7000
GLOBAL_CHAMFER_WEIGHT=750
TARGET_WEIGHT=130
TARGET_RENDER_WEIGHT=8
REG_WEIGHT=1800
N_BUCKETS=12
EPOCHS=7000
SOFT_POINTS=1024
SOFT_SEMANTIC_WEIGHT=0.10
SOFT_TEMPERATURE=0.03
HARD_WEIGHT=0.60
SOFT_WEIGHT=0.40

OUT=./jobs_with_target_guidance/outputs/${LABEL}_partfield_hybrid_${SLURM_JOB_ID:-manual}

echo "======================================================"
echo "H100 dachshund -> pitbull/bulldog PartField hybrid guidance, ${EPOCHS} epochs"
echo "SOURCE=${SOURCE}"
echo "PROMPT=${PROMPT}"
echo "TARGET=${TARGET}"
echo "SOURCE_PARTFIELD_LABELS=${SOURCE_PARTFIELD_LABELS}"
echo "TARGET_PARTFIELD_LABELS=${TARGET_PARTFIELD_LABELS}"
echo "SOURCE_PARTFIELD_FEATURES=${SOURCE_PARTFIELD_FEATURES}"
echo "TARGET_PARTFIELD_FEATURES=${TARGET_PARTFIELD_FEATURES}"
echo "OUT=${OUT}"
echo "HARD_WEIGHT=${HARD_WEIGHT}"
echo "SOFT_WEIGHT=${SOFT_WEIGHT}"
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
  --partfield_guidance_mode hybrid \
  --partfield_hard_weight "${HARD_WEIGHT}" \
  --partfield_soft_weight "${SOFT_WEIGHT}" \
  --partfield_soft_points "${SOFT_POINTS}" \
  --partfield_soft_semantic_weight "${SOFT_SEMANTIC_WEIGHT}" \
  --partfield_soft_temperature "${SOFT_TEMPERATURE}" \
  --regularize_jacobians_weight "${REG_WEIGHT}" \
  --output_path "${OUT}" \
  --epochs "${EPOCHS}"

python jobs_with_target_guidance/evaluate_target_pipeline.py \
  --output-dir "${OUT}" \
  --samples 5000 \
  --part-samples 1000

echo "END_TIME=$(date)"
echo "Finished: ${OUT}"
