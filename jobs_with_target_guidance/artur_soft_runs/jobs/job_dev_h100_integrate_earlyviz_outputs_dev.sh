#!/bin/bash
#SBATCH --job-name=dev_integrate_early
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_integrate_early_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_integrate_early_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

EPOCHS=(2 3 4 5 6 7 8 9 10 20 30 40 50)

copy_epochs() {
  local early="$1"
  local target="$2"

  echo "======================================================"
  echo "EARLY=${early}"
  echo "TARGET=${target}"
  echo "======================================================"

  for epoch in "${EPOCHS[@]}"; do
    printf -v padded "%04d" "${epoch}"

    mkdir -p "${target}/colored_meshes" "${target}/correspondence" "${target}/jacobians" "${target}/displacement_viz"

    cp -f "${early}/colored_meshes/mesh_epoch_${epoch}.ply" \
      "${target}/colored_meshes/mesh_epoch_${epoch}.ply"
    cp -f "${early}/correspondence/correspondence_epoch_${epoch}.json" \
      "${target}/correspondence/correspondence_epoch_${epoch}.json"
    cp -f "${early}/jacobians/jacobians_epoch_${epoch}.npy" \
      "${target}/jacobians/jacobians_epoch_${epoch}.npy"

    cp -f "${early}/displacement_viz/disp_epoch_${padded}.png" \
      "${target}/displacement_viz/disp_epoch_${padded}.png"
    cp -f "${early}/displacement_viz/disp_epoch_${padded}.ply" \
      "${target}/displacement_viz/disp_epoch_${padded}.ply"
    cp -f "${early}/displacement_viz/top_displacements_epoch_${padded}.csv" \
      "${target}/displacement_viz/top_displacements_epoch_${padded}.csv"
  done
}

GLOBAL_EARLY=jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_earlyviz_global_chamfer_only_50ep_4889003
GLOBAL_TARGET=jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_global_chamfer_only_2500ep_4807067

HARD_EARLY=jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_earlyviz_hard_partfield_chamfer_only_50ep_4889002
HARD_TARGET=jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_hard_partfield_chamfer_only_2500ep_4807068

SOFT_EARLY=jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_earlyviz_artur_soft_partfield_chamfer_only_50ep_4889001
SOFT_TARGET=jobs_with_target_guidance/artur_soft_runs/outputs_dev/bulldog_to_dachshund_artur_soft_dev_h100_single_artur_soft_partfield_chamfer_only_2500ep_4807069

copy_epochs "${GLOBAL_EARLY}" "${GLOBAL_TARGET}"
copy_epochs "${HARD_EARLY}" "${HARD_TARGET}"
copy_epochs "${SOFT_EARLY}" "${SOFT_TARGET}"

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/meshup_mplconfig_${SLURM_JOB_ID}
mkdir -p "${MPLCONFIGDIR}"

/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/miniconda3/envs/meshup_new/bin/python \
  jobs_with_target_guidance/visualize_run_displacements.py \
  "${GLOBAL_TARGET}" \
  "${HARD_TARGET}" \
  "${SOFT_TARGET}"

for target in "${GLOBAL_TARGET}" "${HARD_TARGET}" "${SOFT_TARGET}"; do
  echo "Integrated displacement files in: ${target}/displacement_viz"
  find "${target}/displacement_viz" -maxdepth 1 -name 'disp_epoch_*.png' | sort | wc -l
done
