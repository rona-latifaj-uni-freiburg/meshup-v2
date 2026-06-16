#!/bin/bash
#SBATCH --job-name=dev_deform_param
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_deform_param_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_deform_param_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: sbatch $0 PARAMETERIZATION VARIANT_ID JAC_REG_WEIGHT JAC_OUTLIER_WEIGHT [EPOCHS]"
  echo "PARAMETERIZATION: jacobian | vertex"
  echo "VARIANT_ID: 1 hard PartField, 2 Artur soft PartField"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PARAMETERIZATION="$1"
VARIANT_ID="$2"
JAC_REG="$3"
JAC_OUTLIER="$4"
EPOCHS="${5:-${EPOCHS:-500}}"

case "${PARAMETERIZATION}" in
  jacobian|vertex) ;;
  *)
    echo "Unknown PARAMETERIZATION=${PARAMETERIZATION}. Expected jacobian or vertex."
    exit 2
    ;;
esac

case "${VARIANT_ID}" in
  1|2) ;;
  *)
    echo "This comparison uses PartField variants, so use VARIANT_ID 1 or 2."
    exit 2
    ;;
esac

export DEFORMATION_PARAMETERIZATION="${PARAMETERIZATION}"
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-750.0}
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-8000.0}
export JACOBIAN_REG_WEIGHT="${JAC_REG}"
export JACOBIAN_OUTLIER_WEIGHT="${JAC_OUTLIER}"
export JACOBIAN_OUTLIER_POWER=${JACOBIAN_OUTLIER_POWER:-4.0}
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-0.0}
export DEFORMATION_GRAD_CLIP_NORM=${DEFORMATION_GRAD_CLIP_NORM:-0.0}
export TARGET_CHAMFER_WARMUP_EPOCHS=${TARGET_CHAMFER_WARMUP_EPOCHS:-50}
export PARTFIELD_CHAMFER_WARMUP_EPOCHS=${PARTFIELD_CHAMFER_WARMUP_EPOCHS:-50}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-100}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-100}
export EXTRA_LOG_EPOCHS=${EXTRA_LOG_EPOCHS:-"2 3 4 5 6 7 8 9 10 20 30 40 50 60 80 100"}

OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_deformation_param
RUN_TAG=deformation_param_dev_h100_single
export VARIANT_SUFFIX="${PARAMETERIZATION}_jreg${JAC_REG}_jout${JAC_OUTLIER}"

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  "${VARIANT_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
