#!/usr/bin/env bash
#SBATCH --job-name=sam2-mask
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --mail-type=BEGIN

set -euo pipefail

# Usage example:
#   sbatch --partition=<usable_gpu_partition> \
#     --export=ALL,BASE_PATH=/work/dlclarge1/latifajr-mesh_creator_for_meshup,IMAGE_NAME=bugatti-centodieci.jpg \
#     scripts/slurm_sam2_mask.sh

BASE_PATH="${BASE_PATH:-/work/dlclarge1/latifajr-mesh_creator_for_meshup}"
IMAGE_NAME="${IMAGE_NAME:-bugatti-centodieci.jpg}"
CONDA_CMD="${CONDA_EXE:-conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${BASE_PATH}/.conda_envs/sam3d-objects}"
MODEL_ID="${MODEL_ID:-facebook/sam2.1-hiera-large}"
PREFER_CENTER="${PREFER_CENTER:-1}"
POINTS_PER_SIDE="${POINTS_PER_SIDE:-64}"
CROP_N_LAYERS="${CROP_N_LAYERS:-2}"
MAX_SIDE="${MAX_SIDE:-0}"
FAST="${FAST:-0}"
FORCE_CPU="${FORCE_CPU:-0}"

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

CMD=(python scripts/generate_mask_sam2.py --base_path "${BASE_PATH}" --image_name "${IMAGE_NAME}" --model_id "${MODEL_ID}" --points_per_side "${POINTS_PER_SIDE}" --crop_n_layers "${CROP_N_LAYERS}")
if [[ "${PREFER_CENTER}" == "1" ]]; then
  CMD+=(--prefer_center)
fi
if [[ "${MAX_SIDE}" != "0" ]]; then
  CMD+=(--max_side "${MAX_SIDE}")
fi
if [[ "${FAST}" == "1" ]]; then
  CMD+=(--fast)
fi

if [[ "${FORCE_CPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

"${CMD[@]}"
python scripts/prepare_masks.py --base_path "${BASE_PATH}"

echo "[DONE] Mask generated at ${BASE_PATH}/mask/${IMAGE_NAME}"
