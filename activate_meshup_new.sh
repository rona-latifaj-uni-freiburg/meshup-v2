#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MESHUP_CONDA_ROOT="${MESHUP_CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
MESHUP_CONDA_ENV="${MESHUP_CONDA_ENV:-meshup_new}"

if [[ ! -f "${MESHUP_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Conda init script not found: ${MESHUP_CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  echo "Set MESHUP_CONDA_ROOT=/path/to/miniconda3 if the workspace layout changed." >&2
  return 1 2>/dev/null || exit 1
fi

# Initialize conda for non-interactive shells (needed for batch jobs).
source "${MESHUP_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${MESHUP_CONDA_ENV}"
module load devel/cuda/12.8
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRIPT_DIR}/.torch_extensions/py310_cu121_sm80_sm90}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/nvdiffrast:${PYTHONPATH:-}"
export HF_HUB_CACHE=/work/dlclarge1/jesslen-od3d/cache/huggingface
export TRANSFORMERS_CACHE=/work/dlclarge1/jesslen-od3d/cache/transformers
echo "Activated ${MESHUP_CONDA_ENV} from ${MESHUP_CONDA_ROOT} with CUDA module and env vars."
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
