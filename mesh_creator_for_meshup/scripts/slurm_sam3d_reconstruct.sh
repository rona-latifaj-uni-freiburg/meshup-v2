#!/usr/bin/env bash
#SBATCH --job-name=sam3d-recon
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN

set -euo pipefail

# Usage examples:
#   Single image:
#   sbatch --partition=<usable_gpu_partition> \
#     --export=ALL,BASE_PATH=/work/dlclarge1/latifajr-mesh_creator_for_meshup,IMAGE_NAME=bugatti-centodieci.jpg \
#     scripts/slurm_sam3d_reconstruct.sh
#
#   Batch mode:
#   sbatch --partition=<usable_gpu_partition> \
#     --export=ALL,BASE_PATH=/work/dlclarge1/latifajr-mesh_creator_for_meshup,RUN_ALL=1 \
#     scripts/slurm_sam3d_reconstruct.sh

BASE_PATH="${BASE_PATH:-/work/dlclarge1/latifajr-mesh_creator_for_meshup}"
SAM3D_REPO="${SAM3D_REPO:-${BASE_PATH}/sam-3d-objects}"
IMAGE_NAME="${IMAGE_NAME:-bugatti-centodieci.jpg}"
RUN_ALL="${RUN_ALL:-0}"
SEED="${SEED:-42}"
CONDA_CMD="${CONDA_EXE:-conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${BASE_PATH}/.conda_envs/sam3d-objects}"
ROTATE_X_DEG="${ROTATE_X_DEG:-0}"
ROTATE_Y_DEG="${ROTATE_Y_DEG:-0}"
ROTATE_Z_DEG="${ROTATE_Z_DEG:-0}"
CONFIG_REL="${CONFIG_REL:-checkpoints/hf/pipeline_lowmem.yaml}"
SWAP_COND_EMBEDDER="${SWAP_COND_EMBEDDER:-1}"
REQUIRE_MESH="${REQUIRE_MESH:-0}"
DEPTH_FIRST_OFFLOAD="${DEPTH_FIRST_OFFLOAD:-1}"
DECODE_OFFLOAD="${DECODE_OFFLOAD:-1}"

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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SAM3D_SWAP_CONDITION_EMBEDDER="${SWAP_COND_EMBEDDER}"
export SAM3D_DEPTH_FIRST_OFFLOAD="${DEPTH_FIRST_OFFLOAD}"
export SAM3D_DECODE_OFFLOAD="${DECODE_OFFLOAD}"

if ! python -c "import pkg_resources" >/dev/null 2>&1; then
  echo "[INFO] pkg_resources missing. Installing setuptools..."
  pip install --no-cache-dir --prefer-binary "setuptools>=68,<81"
fi

cd "${BASE_PATH}"

if [[ ! -f "${SAM3D_REPO}/${CONFIG_REL}" ]]; then
  echo "[ERROR] Missing SAM3D config at ${SAM3D_REPO}/${CONFIG_REL}"
  exit 1
fi

if ! python -c "import gsplat" >/dev/null 2>&1; then
  echo "[INFO] gsplat not found. Installing prebuilt wheel..."
  pip install --no-cache-dir --prefer-binary "gsplat==1.5.3"
fi

COMMON_ARGS=(
  --base_path "${BASE_PATH}"
  --sam3d_repo "${SAM3D_REPO}"
  --config "${CONFIG_REL}"
  --seed "${SEED}"
  --rotate_x_deg "${ROTATE_X_DEG}"
  --rotate_y_deg "${ROTATE_Y_DEG}"
  --rotate_z_deg "${ROTATE_Z_DEG}"
)

if [[ "${REQUIRE_MESH}" == "1" ]]; then
  COMMON_ARGS+=(--require_mesh)
fi

if [[ "${RUN_ALL}" == "1" ]]; then
  python process_image.py "${COMMON_ARGS[@]}" --all
else
  python process_image.py "${COMMON_ARGS[@]}" --image_name "${IMAGE_NAME}"
fi

echo "[DONE] Outputs in ${BASE_PATH}/sam3D/{meta,splat,mesh}"
