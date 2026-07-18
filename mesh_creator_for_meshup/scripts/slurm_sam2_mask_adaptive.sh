#!/usr/bin/env bash
#SBATCH --job-name=sam2-mask-adapt
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --mail-type=BEGIN

set -euo pipefail

BASE_PATH="${BASE_PATH:-/work/dlclarge1/latifajr-mesh_creator_for_meshup}"
IMAGE_NAME="${IMAGE_NAME:-bugatti-centodieci.jpg}"
CONDA_CMD="${CONDA_EXE:-conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${BASE_PATH}/.conda_envs/sam3d-objects}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE_PATH}/.cache}"
export HF_HOME="${HF_HOME:-${BASE_PATH}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${BASE_PATH}/.cache/warp}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${BASE_PATH}/.cache/cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${BASE_PATH}/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-${BASE_PATH}/.cache/torch}"

mkdir -p "${BASE_PATH}/logs" "${XDG_CACHE_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${WARP_CACHE_PATH}" "${CUDA_CACHE_PATH}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
CONDA_BASE="${CONDA_BASE:-$("${CONDA_CMD}" info --base)}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV_PREFIX}"
set -u
cd "${BASE_PATH}"

run_mask() {
  local model_id="$1"
  local pps="$2"
  local crops="$3"
  local force_cpu="$4"
  local max_side="$5"

  if [[ "${force_cpu}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi

  python scripts/generate_mask_sam2.py \
    --base_path "${BASE_PATH}" \
    --image_name "${IMAGE_NAME}" \
    --model_id "${model_id}" \
    --points_per_side "${pps}" \
    --crop_n_layers "${crops}" \
    --max_side "${max_side}" \
    --prefer_center
}

echo "[INFO] Attempt 1: high-quality SAM2 mask on GPU"
if run_mask "facebook/sam2.1-hiera-large" "64" "2" "0" "0"; then
  echo "[OK] GPU HQ mask succeeded"
else
  echo "[WARN] GPU HQ mask failed; retrying on CPU with safer settings"
  run_mask "facebook/sam2.1-hiera-small" "32" "1" "1" "1536"
  echo "[OK] CPU fallback mask succeeded"
fi

python scripts/prepare_masks.py --base_path "${BASE_PATH}"
echo "[DONE] Mask generated at ${BASE_PATH}/mask/${IMAGE_NAME}"
