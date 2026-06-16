#!/bin/bash
#SBATCH --job-name=dev_nd_anim
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_nd_anim_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_nd_anim_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch $0 PAIR_ID MODE"
  echo "PAIR_ID: 0 bulldog->horse, 1 bulldog->cat, 2 bulldog->golden, 3 bulldog->bear, 4 dachshund->golden"
  echo "MODE: pf_chamfer | pf_chamfer_jneighbor"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PAIR_ID="$1"
MODE="$2"
EPOCHS=${EPOCHS:-4000}
JNEIGHBOR_WEIGHT=${JNEIGHBOR_WEIGHT:-1800}
RUN_TAG=${RUN_TAG:-dev_h100_no_dino_animals_refhard_single}
OUTPUT_SERIES=${OUTPUT_SERIES:-outputs_dev_no_dino_animals_4000_fixed}

case "${MODE}" in
  pf_chamfer)
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/${OUTPUT_SERIES}/no_regularizer
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=0.0
    ;;
  pf_chamfer_jneighbor)
    OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/${OUTPUT_SERIES}/jneighbor_${JNEIGHBOR_WEIGHT}
    export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT="${JNEIGHBOR_WEIGHT}"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Expected pf_chamfer or pf_chamfer_jneighbor."
    exit 2
    ;;
esac

export DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-jacobian}
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-0.0}
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-8000.0}
export JACOBIAN_REG_WEIGHT=${JACOBIAN_REG_WEIGHT:-0.0}
export JACOBIAN_OUTLIER_WEIGHT=${JACOBIAN_OUTLIER_WEIGHT:-0.0}
export TARGET_CHAMFER_WARMUP_EPOCHS=${TARGET_CHAMFER_WARMUP_EPOCHS:-0}
export PARTFIELD_CHAMFER_WARMUP_EPOCHS=${PARTFIELD_CHAMFER_WARMUP_EPOCHS:-0}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-250}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-250}
export EXTRA_LOG_EPOCHS=${EXTRA_LOG_EPOCHS:-"1 10 20 30 40 50 60 70 80 90 100"}

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_no_dino_animal_pair_4000.sh \
  "${PAIR_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}" \
  "${MODE}"
