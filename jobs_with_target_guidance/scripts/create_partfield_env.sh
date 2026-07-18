#!/bin/bash
set -euo pipefail

ENV_NAME=${PARTFIELD_ENV:-partfield}
PYTHON_VERSION=${PARTFIELD_PYTHON_VERSION:-3.10}
TORCH_VERSION=${PARTFIELD_TORCH_VERSION:-2.4.0}
TORCHVISION_VERSION=${PARTFIELD_TORCHVISION_VERSION:-0.19.0}
TORCHAUDIO_VERSION=${PARTFIELD_TORCHAUDIO_VERSION:-2.4.0}
CUDA_TAG=${PARTFIELD_CUDA_TAG:-cu124}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESHUP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${MESHUP_ROOT}/../.." && pwd)"
MESHUP_CONDA_ROOT="${MESHUP_CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"

if [[ ! -f "${MESHUP_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Conda init script not found: ${MESHUP_CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  echo "Set MESHUP_CONDA_ROOT=/path/to/miniconda3 if the workspace layout changed." >&2
  exit 1
fi

source "${MESHUP_CONDA_ROOT}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; updating pip packages."
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install "setuptools<81" wheel

python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

python -m pip install \
  psutil \
  lightning==2.2 \
  h5py \
  yacs \
  trimesh \
  scikit-image \
  loguru \
  boto3 \
  mesh2sdf \
  tetgen \
  pymeshlab \
  plyfile \
  einops \
  libigl \
  polyscope \
  potpourri3d \
  simple_parsing \
  arrgh \
  open3d \
  vtk

python -m pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

python - <<'PY'
import torch
print("PartField env ready")
print("python ok")
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
PY
