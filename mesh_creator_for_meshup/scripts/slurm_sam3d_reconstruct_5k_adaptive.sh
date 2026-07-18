#!/usr/bin/env bash
#SBATCH --job-name=sam3d-recon-5k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN

set -euo pipefail

BASE_PATH="${BASE_PATH:-/work/dlclarge1/latifajr-mesh_creator_for_meshup}"
SAM3D_REPO="${SAM3D_REPO:-${BASE_PATH}/sam-3d-objects}"
IMAGE_NAME="${IMAGE_NAME:-bugatti-centodieci.jpg}"
SEED="${SEED:-42}"
CONDA_CMD="${CONDA_EXE:-conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${BASE_PATH}/.conda_envs/sam3d-objects}"
ROTATE_X_DEG="${ROTATE_X_DEG:-0}"
ROTATE_Y_DEG="${ROTATE_Y_DEG:-0}"
ROTATE_Z_DEG="${ROTATE_Z_DEG:-0}"
SWAP_COND_EMBEDDER="${SWAP_COND_EMBEDDER:-1}"
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

cd "${BASE_PATH}"

if ! python -c "import pkg_resources" >/dev/null 2>&1; then
  pip install --no-cache-dir --prefer-binary "setuptools>=68,<81"
fi
if ! python -c "import gsplat" >/dev/null 2>&1; then
  pip install --no-cache-dir --prefer-binary "gsplat==1.5.3"
fi

run_recon() {
  local config_rel="$1"
  python process_image.py \
    --base_path "${BASE_PATH}" \
    --sam3d_repo "${SAM3D_REPO}" \
    --config "${config_rel}" \
    --seed "${SEED}" \
    --rotate_x_deg "${ROTATE_X_DEG}" \
    --rotate_y_deg "${ROTATE_Y_DEG}" \
    --rotate_z_deg "${ROTATE_Z_DEG}" \
    --require_mesh \
    --image_name "${IMAGE_NAME}"
}

extract_largest_component() {
  local mesh_path="$1"
  if [[ -f "$mesh_path" ]]; then
    echo "[INFO] Extracting largest connected component from $mesh_path"
    python "${BASE_PATH}/scripts/extract_largest_component.py" "$mesh_path"
  fi
}

echo "[INFO] Processing: ${IMAGE_NAME}"
echo "[INFO] Attempt 1: 5k-vertex mesh config"
if run_recon "checkpoints/hf/pipeline_mesh_5k.yaml"; then
  echo "[INFO] 5k mesh succeeded"
  mesh_out="${BASE_PATH}/sam3D/mesh/${IMAGE_NAME%.*}.ply"
  extract_largest_component "$mesh_out"
  echo "[DONE] Successfully processed ${IMAGE_NAME} (5k config)"
  exit 0
fi

echo "[WARN] 5k config failed; attempting lowmem config"
if run_recon "checkpoints/hf/pipeline_mesh_lowmem.yaml"; then
  echo "[INFO] Low-memory mesh succeeded"
  mesh_out="${BASE_PATH}/sam3D/mesh/${IMAGE_NAME%.*}.ply"
  extract_largest_component "$mesh_out"
  echo "[DONE] Successfully processed ${IMAGE_NAME} (lowmem config)"
  exit 0
fi

echo "[ERROR] All configs failed for ${IMAGE_NAME}"
exit 1
