#!/usr/bin/env bash
#SBATCH --job-name=sam2-mask-array
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail

BASE_PATH="${BASE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE_LIST="${IMAGE_LIST:-${BASE_PATH}/dense_images_manifest.txt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
LINE_NO=$((TASK_ID + 1))
IMAGE_NAME="$(sed -n "${LINE_NO}p" "${IMAGE_LIST}")"

if [[ -z "${IMAGE_NAME}" ]]; then
  echo "[ERROR] No image name for array task ${TASK_ID} in ${IMAGE_LIST}"
  exit 1
fi

mkdir -p "${BASE_PATH}/image" "${BASE_PATH}/mask" "${BASE_PATH}/logs"

if [[ -f "${BASE_PATH}/mask/${IMAGE_NAME}" ]]; then
  echo "[INFO] Reusing existing mask for ${IMAGE_NAME}"
  exit 0
fi

export BASE_PATH IMAGE_NAME
bash "${BASE_PATH}/scripts/slurm_sam2_mask_adaptive.sh"
