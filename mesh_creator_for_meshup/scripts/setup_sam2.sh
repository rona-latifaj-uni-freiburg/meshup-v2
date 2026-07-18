#!/usr/bin/env bash
set -euo pipefail

# Install SAM2 into the existing sam3d-objects env for image mask generation.
# Usage:
#   bash scripts/setup_sam2.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM2_REPO="${ROOT_DIR}/sam2"
CONDA_CMD="${CONDA_EXE:-conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${ROOT_DIR}/.conda_envs/sam3d-objects}"

if [[ ! -d "${SAM2_REPO}" ]]; then
  echo "[ERROR] Missing SAM2 repo at ${SAM2_REPO}"
  echo "Clone it first: git clone https://github.com/facebookresearch/sam2.git ${SAM2_REPO}"
  exit 1
fi

# shellcheck disable=SC1091
source "$("${CONDA_CMD}" info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV_PREFIX}"
set -u

# Build-free install is usually enough for image masks and avoids CUDA extension issues.
export SAM2_BUILD_CUDA=0
pip install --no-cache-dir --prefer-binary -e "${SAM2_REPO}"
pip install --no-cache-dir --prefer-binary huggingface_hub pillow-avif-plugin

echo "[DONE] SAM2 installed in ${CONDA_ENV_PREFIX}"
