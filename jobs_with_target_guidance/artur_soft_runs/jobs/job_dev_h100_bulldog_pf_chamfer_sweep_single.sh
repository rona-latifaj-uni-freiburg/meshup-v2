#!/bin/bash
#SBATCH --job-name=dev_bdog_pfch
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_bdog_pfch_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_bdog_pfch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch $0 MODE VARIANT_ID [WEIGHT]"
  echo "MODE: dino | reg | jneighbor | contain | contain_edge | balanced | jump | asym_jump | robust_jump | unbalanced_jump | mutual_jump"
  echo "VARIANT_ID: 1 hard PartField, 2 Artur soft PartField"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

MODE="$1"
VARIANT_ID="$2"
WEIGHT="${3:-0}"
EPOCHS=${EPOCHS:-2500}

case "${VARIANT_ID}" in
  1|2) ;;
  *)
    echo "This sweep combines global Chamfer with PartField, so use VARIANT_ID 1 or 2."
    exit 2
    ;;
esac

export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-750.0}
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-8000.0}

case "${MODE}" in
  dino)
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_dino
    RUN_TAG=artur_pf_chamfer_dino_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_dino
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-0.0}
    export ENABLE_TARGET_DINO_GUIDANCE=1
    ;;
  reg)
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_reg_${WEIGHT}
    RUN_TAG=artur_pf_chamfer_reg${WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_reg${WEIGHT}
    export JACOBIAN_REG_WEIGHT="${WEIGHT}"
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=0.0
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  jneighbor)
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jneighbor_${WEIGHT}
    RUN_TAG=artur_pf_chamfer_jneighbor${WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor${WEIGHT}
    export JACOBIAN_REG_WEIGHT=0.0
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT="${WEIGHT}"
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  contain)
    CONTAINMENT_WEIGHT="${WEIGHT:-10}"
    if [[ "${CONTAINMENT_WEIGHT}" == "0" ]]; then
      CONTAINMENT_WEIGHT=10
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_containment_${CONTAINMENT_WEIGHT}
    RUN_TAG=artur_pf_chamfer_contain${CONTAINMENT_WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_contain${CONTAINMENT_WEIGHT}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-1000.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-250.0}
    export JACOBIAN_OUTLIER_POWER=${JACOBIAN_OUTLIER_POWER:-4.0}
    export PARTFIELD_CONTAINMENT_WEIGHT="${CONTAINMENT_WEIGHT}"
    export PARTFIELD_CONTAINMENT_MARGIN=${PARTFIELD_CONTAINMENT_MARGIN:-0.02}
    export PARTFIELD_CONTAINMENT_MAX_WEIGHT=${PARTFIELD_CONTAINMENT_MAX_WEIGHT:-2.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  contain_edge)
    CONTAINMENT_WEIGHT="${WEIGHT:-10}"
    if [[ "${CONTAINMENT_WEIGHT}" == "0" ]]; then
      CONTAINMENT_WEIGHT=10
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_containment_edge_${CONTAINMENT_WEIGHT}
    RUN_TAG=artur_pf_chamfer_contain${CONTAINMENT_WEIGHT}_edge_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_contain${CONTAINMENT_WEIGHT}_edge
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-1000.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-250.0}
    export JACOBIAN_OUTLIER_POWER=${JACOBIAN_OUTLIER_POWER:-4.0}
    export PARTFIELD_CONTAINMENT_WEIGHT="${CONTAINMENT_WEIGHT}"
    export PARTFIELD_CONTAINMENT_MARGIN=${PARTFIELD_CONTAINMENT_MARGIN:-0.02}
    export PARTFIELD_CONTAINMENT_MAX_WEIGHT=${PARTFIELD_CONTAINMENT_MAX_WEIGHT:-2.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-250.0}
    export EDGE_STRETCH_THRESHOLD=${EDGE_STRETCH_THRESHOLD:-1.5}
    export EDGE_STRETCH_MAX_WEIGHT=${EDGE_STRETCH_MAX_WEIGHT:-1.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  balanced)
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_balanced
    RUN_TAG=artur_pf_chamfer_balanced_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_balanced_sinkhorn
    export PARTFIELD_GUIDANCE_MODE=balanced
    export PARTFIELD_HARD_WEIGHT=1.0
    export PARTFIELD_SOFT_WEIGHT=0.0
    export PARTFIELD_BALANCED_SINKHORN_ITERS=${PARTFIELD_BALANCED_SINKHORN_ITERS:-20}
    export ARTUR_SOFT_MATCH_SPACE=${ARTUR_SOFT_MATCH_SPACE:-hybrid}
    export ARTUR_SOFT_GEOMETRY_SIGMA=${ARTUR_SOFT_GEOMETRY_SIGMA:-0.5}
    export ARTUR_SOFT_SEMANTIC_WEIGHT=${ARTUR_SOFT_SEMANTIC_WEIGHT:-1.0}
    export ARTUR_SOFT_TEMPERATURE=${ARTUR_SOFT_TEMPERATURE:-0.08}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
    export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  jump)
    JUMP_WEIGHT="${WEIGHT:-500}"
    if [[ "${JUMP_WEIGHT}" == "0" ]]; then
      JUMP_WEIGHT=500
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jump_${JUMP_WEIGHT}
    RUN_TAG=artur_pf_chamfer_jump${JUMP_WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_jump${JUMP_WEIGHT}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
    export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
    export EDGE_DISPLACEMENT_JUMP_WEIGHT="${JUMP_WEIGHT}"
    export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
    export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  asym_jump)
    JUMP_WEIGHT="${WEIGHT:-500}"
    if [[ "${JUMP_WEIGHT}" == "0" ]]; then
      JUMP_WEIGHT=500
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_${JUMP_WEIGHT}
    RUN_TAG=artur_pf_chamfer_asymjump${JUMP_WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_asym035_jump${JUMP_WEIGHT}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
    export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
    export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT:-0.35}
    export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT:-1.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
    export EDGE_DISPLACEMENT_JUMP_WEIGHT="${JUMP_WEIGHT}"
    export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
    export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  robust_jump)
    JUMP_WEIGHT="${WEIGHT:-500}"
    if [[ "${JUMP_WEIGHT}" == "0" ]]; then
      JUMP_WEIGHT=500
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_robust_jump_${JUMP_WEIGHT}
    RUN_TAG=artur_pf_chamfer_robustjump${JUMP_WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_robustt2sauto_jump${JUMP_WEIGHT}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
    export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
    export PARTFIELD_TGT_TO_SRC_ROBUST_SCALE=${PARTFIELD_TGT_TO_SRC_ROBUST_SCALE:--1.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
    export EDGE_DISPLACEMENT_JUMP_WEIGHT="${JUMP_WEIGHT}"
    export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
    export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  unbalanced_jump)
    JUMP_WEIGHT="${WEIGHT:-500}"
    if [[ "${JUMP_WEIGHT}" == "0" ]]; then
      JUMP_WEIGHT=500
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_unbalanced_jump_${JUMP_WEIGHT}
    RUN_TAG=artur_pf_chamfer_unbalancedjump${JUMP_WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_unbalanced020_rho030_jump${JUMP_WEIGHT}
    export PARTFIELD_GUIDANCE_MODE=hard_unbalanced
    export PARTFIELD_HARD_WEIGHT=1.0
    export PARTFIELD_SOFT_WEIGHT=0.0
    export PARTFIELD_UNBALANCED_TRANSPORT_WEIGHT=${PARTFIELD_UNBALANCED_TRANSPORT_WEIGHT:-0.20}
    export PARTFIELD_UNBALANCED_TRANSPORT_RHO=${PARTFIELD_UNBALANCED_TRANSPORT_RHO:-0.30}
    export PARTFIELD_BALANCED_SINKHORN_ITERS=${PARTFIELD_BALANCED_SINKHORN_ITERS:-20}
    export ARTUR_SOFT_MATCH_SPACE=${ARTUR_SOFT_MATCH_SPACE:-hybrid}
    export ARTUR_SOFT_GEOMETRY_SIGMA=${ARTUR_SOFT_GEOMETRY_SIGMA:-0.5}
    export ARTUR_SOFT_SEMANTIC_WEIGHT=${ARTUR_SOFT_SEMANTIC_WEIGHT:-1.0}
    export ARTUR_SOFT_TEMPERATURE=${ARTUR_SOFT_TEMPERATURE:-0.04}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
    export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
    export EDGE_DISPLACEMENT_JUMP_WEIGHT="${JUMP_WEIGHT}"
    export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
    export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  mutual_jump)
    JUMP_WEIGHT="${WEIGHT:-500}"
    if [[ "${JUMP_WEIGHT}" == "0" ]]; then
      JUMP_WEIGHT=500
    fi
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_mutual_jump_${JUMP_WEIGHT}
    RUN_TAG=artur_pf_chamfer_mutualjump${JUMP_WEIGHT}_dev_h100_single
    export VARIANT_SUFFIX=pf_chamfer_jneighbor1000_mutual015_jump${JUMP_WEIGHT}
    export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
    export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
    export PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=${PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT:-0.15}
    export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}
    export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-0.0}
    export EDGE_DISPLACEMENT_JUMP_WEIGHT="${JUMP_WEIGHT}"
    export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
    export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-2.0}
    export ENABLE_TARGET_DINO_GUIDANCE=0
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected dino, reg, jneighbor, contain, contain_edge, balanced, jump, asym_jump, robust_jump, unbalanced_jump, or mutual_jump."
    exit 2
    ;;
esac

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  "${VARIANT_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
