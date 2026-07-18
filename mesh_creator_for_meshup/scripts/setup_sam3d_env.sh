#!/usr/bin/env bash
set -euo pipefail

# Reproducible setup for facebookresearch/sam-3d-objects inside this workspace.
# Usage:
#   bash scripts/setup_sam3d_env.sh
# Optional:
#   export CONDA_EXE=mamba   # default: conda
#   export CONDA_ENV_PREFIX=/path/to/env  # default: .conda_envs/sam3d-objects
#   export SKIP_HF_DOWNLOAD=1
#   export SKIP_BLENDER=1     # default: 1, avoids bpy wheel issue
#   export INSTALL_FLASH_ATTN=0  # default: 0, enable only on compatible GPU/toolchain
#   export INSTALL_GSPLAT=1      # default: 1, install the PyPI wheel used by jobs
#   export INSTALL_SAM2=1        # default: 1, install local SAM2 checkout for masks

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3D_REPO="${ROOT_DIR}/sam-3d-objects"
SAM2_REPO="${ROOT_DIR}/sam2"
CONDA_CMD="${CONDA_EXE:-conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${ROOT_DIR}/.conda_envs/sam3d-objects}"

if [[ ! -d "${SAM3D_REPO}" ]]; then
  echo "[ERROR] Missing repo: ${SAM3D_REPO}"
  echo "Clone it first: git clone https://github.com/facebookresearch/sam-3d-objects.git ${SAM3D_REPO}"
  exit 1
fi

if ! command -v "${CONDA_CMD}" >/dev/null 2>&1; then
  echo "[ERROR] ${CONDA_CMD} is not installed or not in PATH"
  exit 1
fi

CONDA_BASE="$("${CONDA_CMD}" info --base)"
mkdir -p "$(dirname "${CONDA_ENV_PREFIX}")" "${ROOT_DIR}/.conda_pkgs" "${ROOT_DIR}/.cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${ROOT_DIR}/.cache/warp}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${ROOT_DIR}/.cache/cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/.cache/matplotlib}"
export TORCH_HOME="${TORCH_HOME:-${ROOT_DIR}/.cache/torch}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-false}"
export CONDA_SOLVER="${CONDA_SOLVER:-libmamba}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${ROOT_DIR}/.conda_pkgs:${CONDA_BASE}/pkgs}"

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

cd "${SAM3D_REPO}"

ENV_YML="environments/default.yml"
if ! grep -qx "  - nodefaults" "${ENV_YML}"; then
  ENV_YML="${ROOT_DIR}/.cache/default-nodefaults.yml"
  awk '{print} $0=="  - conda-forge"{print "  - nodefaults"}' environments/default.yml > "${ENV_YML}"
fi

if [[ ! -d "${CONDA_ENV_PREFIX}/conda-meta" ]]; then
  echo "[INFO] Creating local conda env at ${CONDA_ENV_PREFIX}"
  "${CONDA_CMD}" env create -p "${CONDA_ENV_PREFIX}" -f "${ENV_YML}"
else
  echo "[INFO] Reusing local conda env at ${CONDA_ENV_PREFIX}"
fi

set +u
conda activate "${CONDA_ENV_PREFIX}"
set -u

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[INFO] nvcc not found in env; installing CUDA compiler/runtime from NVIDIA channel"
  if ! "${CONDA_CMD}" install -p "${CONDA_ENV_PREFIX}" -y --override-channels -c nvidia -c conda-forge cuda-nvcc=12.1 cuda-cudart-dev=12.1; then
    echo "[WARN] Failed to install cuda-nvcc/cuda-cudart-dev from nvidia channel; trying conda-forge fallback"
    "${CONDA_CMD}" install -p "${CONDA_ENV_PREFIX}" -y --override-channels -c conda-forge cudatoolkit-dev=12.1
  fi
fi

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

# On login/headless nodes torch may detect no visible CUDA devices, which causes
# cpp_extension to produce an empty arch list and fail builds (IndexError).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] nvcc still not found after CUDA toolkit install."
  echo "[ERROR] Cannot build gsplat without CUDA compiler."
  exit 1
fi

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
export PIP_NO_CACHE_DIR=1
PIP_FLAGS=(--no-cache-dir --prefer-binary)
TORCH_VERSION="2.5.1+cu121"
TORCHVISION_VERSION="0.20.1+cu121"
TORCHAUDIO_VERSION="2.5.1+cu121"

echo "[INFO] Installing sam-3d-objects dependencies"
SKIP_BLENDER="${SKIP_BLENDER:-1}"
INSTALL_PROFILE="${INSTALL_PROFILE:-inference}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
INSTALL_GSPLAT="${INSTALL_GSPLAT:-1}"
INSTALL_SAM2="${INSTALL_SAM2:-1}"

if [[ "${INSTALL_PROFILE}" == "inference" ]]; then
  echo "[INFO] Using focused inference install profile"

  # Install torch first so build-time imports in PyTorch3D and SAM2 work.
  pip install "${PIP_FLAGS[@]}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

  pip install "${PIP_FLAGS[@]}" -e . --no-deps

  pip install "${PIP_FLAGS[@]}" \
    "numpy==1.26.4" \
    "pillow" \
    "pyyaml" \
    "omegaconf==2.3.0" \
    "hydra-core==1.3.2" \
    "loguru==0.7.2" \
    "tqdm" \
    "safetensors" \
    "easydict==1.13" \
    "scipy==1.13.1" \
    "trimesh" \
    "matplotlib" \
    "plyfile" \
    "iopath>=0.1.10" \
    "fvcore==0.1.5.post20221221" \
    "optree==0.14.1" \
    "einops" \
    "einops-exts==0.0.4" \
    "lightning==2.3.3" \
    "spconv-cu121==2.3.8" \
    "opencv-python==4.9.0.80" \
    "pycocotools==2.0.7" \
    "huggingface_hub<1.0" \
    "timm==0.9.16" \
    "transformers<5" \
    "roma==1.5.1"

  pip install "${PIP_FLAGS[@]}" --no-deps "open3d==0.18.0"
  pip install "${PIP_FLAGS[@]}" --no-deps \
    "utils3d @ https://github.com/EasternJournalist/utils3d/archive/3913c65d81e05e47b9f367250cf8c0f7462a0900.zip"
  pip install "${PIP_FLAGS[@]}" \
    "astor==0.8.1" \
    "igraph==0.11.8" \
    "xatlas==0.0.9" \
    "pymeshfix==0.17.0" \
    "dash>=2.6.0" \
    "nbformat>=5.7.0" \
    "configargparse" \
    "addict" \
    "pandas>=1.0" \
    "scikit-learn>=0.21" \
    "pyquaternion"

  pip install "${PIP_FLAGS[@]}" --no-build-isolation \
    "pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/75ebeeaea0908c5527e7b1e305fbc7681382db47.zip"

  pip install "${PIP_FLAGS[@]}" --no-deps \
    "MoGe @ https://github.com/microsoft/MoGe/archive/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b.zip"

  if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    pip install "${PIP_FLAGS[@]}" --no-build-isolation flash_attn==2.8.3
  else
    echo "[INFO] Skipping flash_attn install (INSTALL_FLASH_ATTN=0; SDPA is the default backend)"
  fi

  pip install "${PIP_FLAGS[@]}" kaolin==0.17.0

  if [[ "${INSTALL_GSPLAT}" == "1" ]]; then
    pip install "${PIP_FLAGS[@]}" "gsplat==1.5.3"
  else
    echo "[INFO] Skipping gsplat install (INSTALL_GSPLAT=0)"
    echo "[INFO] Install later: INSTALL_GSPLAT=1 SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh"
  fi
elif [[ "${SKIP_BLENDER}" == "1" ]]; then
  echo "[INFO] Using blender-free install profile (skipping bpy)"

  TMP_REQUIREMENTS="$(mktemp)"
  trap 'rm -f "${TMP_REQUIREMENTS}"' EXIT

  # bpy is optional for this image+mask -> mesh pipeline and is often unavailable.
  # nvidia-pyindex is also optional here because indexes are already set via env vars.
  # MoGe is installed from a GitHub archive below to avoid invoking system git
  # from inside the CUDA-heavy conda environment.
  grep -vE '^(bpy==4\.3\.0|nvidia-pyindex==1\.0\.9|MoGe @ git\+https://github\.com/microsoft/MoGe\.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b)$' requirements.txt > "${TMP_REQUIREMENTS}"

  pip install "${PIP_FLAGS[@]}" -r "${TMP_REQUIREMENTS}"
  pip install "${PIP_FLAGS[@]}" -e . --no-deps

  # Install torch first, then install pytorch3d without build isolation.
  # Pytorch3D setup imports torch at build-time and may fail in isolated build envs.
  pip install "${PIP_FLAGS[@]}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

  pip install "${PIP_FLAGS[@]}" --no-build-isolation \
    "pytorch3d @ https://github.com/facebookresearch/pytorch3d/archive/75ebeeaea0908c5527e7b1e305fbc7681382db47.zip"

  pip install "${PIP_FLAGS[@]}" --no-deps \
    "MoGe @ https://github.com/microsoft/MoGe/archive/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b.zip"

  if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    # flash_attn may require specific CUDA/compiler compatibility.
    pip install "${PIP_FLAGS[@]}" --no-build-isolation flash_attn==2.8.3
  else
    echo "[INFO] Skipping flash_attn install (INSTALL_FLASH_ATTN=0)"
  fi

  # Install inference dependencies explicitly.
  pip install "${PIP_FLAGS[@]}" kaolin==0.17.0 seaborn==0.13.2 gradio==5.49.0

  if [[ "${INSTALL_GSPLAT}" == "1" ]]; then
    pip install "${PIP_FLAGS[@]}" "gsplat==1.5.3"
  else
    echo "[INFO] Skipping gsplat install (INSTALL_GSPLAT=0)"
    echo "[INFO] Install later: INSTALL_GSPLAT=1 SKIP_HF_DOWNLOAD=1 bash scripts/setup_sam3d_env.sh"
  fi
else
  pip install "${PIP_FLAGS[@]}" -e '.[dev]'
  pip install "${PIP_FLAGS[@]}" -e '.[p3d]'
  pip install "${PIP_FLAGS[@]}" -e '.[inference]'
fi

if [[ "${INSTALL_SAM2}" == "1" ]]; then
  if [[ ! -d "${SAM2_REPO}" ]]; then
    echo "[ERROR] Missing SAM2 repo at ${SAM2_REPO}"
    exit 1
  fi
  echo "[INFO] Installing local SAM2 checkout for mask generation"
  export SAM2_BUILD_CUDA=0
  pip install "${PIP_FLAGS[@]}" -e "${SAM2_REPO}"
  pip install "${PIP_FLAGS[@]}" huggingface_hub pillow-avif-plugin
else
  echo "[INFO] Skipping SAM2 install (INSTALL_SAM2=0)"
fi

if [[ -x ./patching/hydra ]]; then
  echo "[INFO] Applying hydra patch"
  ./patching/hydra
else
  echo "[WARN] patching/hydra script not found or not executable"
fi

if [[ "${SKIP_HF_DOWNLOAD:-0}" != "1" ]]; then
  echo "[INFO] Installing HF CLI and downloading checkpoints (requires approved access)"
  pip install "${PIP_FLAGS[@]}" 'huggingface-hub[cli]<1.0'

  if ! command -v hf >/dev/null 2>&1; then
    echo "[ERROR] hf CLI not found after installation"
    exit 1
  fi

  echo "[INFO] If not already authenticated, run: hf auth login"

  TAG=hf
  hf download \
    --repo-type model \
    --local-dir checkpoints/${TAG}-download \
    --max-workers 1 \
    facebook/sam-3d-objects

  mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
  rm -rf checkpoints/${TAG}-download
else
  echo "[INFO] Skipping checkpoint download because SKIP_HF_DOWNLOAD=1"
fi

echo "[DONE] Environment setup complete"
echo "[NEXT] Activate env with: conda activate ${CONDA_ENV_PREFIX}"
