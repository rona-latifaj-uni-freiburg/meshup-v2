#!/bin/bash
#SBATCH --job-name=dev_artur_early
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_artur_early_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_artur_early_%j.err

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 VARIANT_ID"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

VARIANT_ID="$1"
EPOCHS=50
OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev
RUN_TAG=artur_soft_dev_h100_single_earlyviz
EXTRA_LOG_EPOCHS="2 3 4 5 6 7 8 9 10 20 30 40 50"
export EXTRA_LOG_EPOCHS

case "${VARIANT_ID}" in
  0) VARIANT=global_chamfer_only ;;
  1) VARIANT=hard_partfield_chamfer_only ;;
  2) VARIANT=artur_soft_partfield_chamfer_only ;;
  *)
    echo "Unknown VARIANT_ID=${VARIANT_ID}. Expected 0, 1, or 2."
    exit 1
    ;;
esac

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  "${VARIANT_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"

OUT="${OUTPUT_ROOT}/bulldog_to_dachshund_${RUN_TAG}_${VARIANT}_${EPOCHS}ep_${SLURM_JOB_ID}"

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/meshup_mplconfig_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python \
  jobs_with_target_guidance/visualize_run_displacements.py \
  "${OUT}" \
  --epochs 1 2 3 4 5 6 7 8 9 10 20 30 40 50

echo "EARLY_VIZ_OUT=${OUT}"
