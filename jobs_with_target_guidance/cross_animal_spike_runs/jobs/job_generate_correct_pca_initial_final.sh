#!/bin/bash
#SBATCH --job-name=cross_animal_pca_if
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/correct_pca_initial_final_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/correct_pca_initial_final_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
source ./activate_meshup_new.sh

OUTPUT_ROOT=${OUTPUT_ROOT:-jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_vcorr_dense_pca}
MODEL=${MODEL:-dinov2_vitl14}
IMAGE_SIZE=${IMAGE_SIZE:-518}
CROP_MARGIN_RATIO=${CROP_MARGIN_RATIO:-0.04}
MIN_CROP_SIDE=${MIN_CROP_SIDE:-256}

find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r run_dir; do
  epoch_dir="${run_dir}/epoch_renders"
  [[ -d "${epoch_dir}/epoch_00001" ]] || continue

  last_epoch=$(
    find "${epoch_dir}" -mindepth 1 -maxdepth 1 -type d -name 'epoch_*' \
      | sed 's/.*epoch_//' \
      | sort -n \
      | tail -n 1
  )

  if [[ -z "${last_epoch}" || "${last_epoch}" == "00001" ]]; then
    echo "Skipping ${run_dir}: no final epoch render found"
    continue
  fi

  last_epoch_num=$((10#${last_epoch}))
  echo "Generating cropped combined PCA for ${run_dir}: epochs 1 and ${last_epoch_num}"

  python generate_pca_evolution_4views.py \
    --epoch_renders_dir "${epoch_dir}" \
    --output_dir "${run_dir}/pca_initial_final" \
    --epochs "1,${last_epoch_num}" \
    --test_image_pca_script test_image_pca.py \
    --model "${MODEL}" \
    --image_size "${IMAGE_SIZE}" \
    --crop_margin_ratio "${CROP_MARGIN_RATIO}" \
    --min_crop_side "${MIN_CROP_SIDE}"
done

echo "Done generating corrected initial/final PCA under ${OUTPUT_ROOT}"
