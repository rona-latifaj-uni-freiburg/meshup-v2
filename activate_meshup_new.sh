#!/usr/bin/env bash
set -e
conda activate meshup_new
module load devel/cuda/12.8
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_HUB_CACHE=/work/dlclarge1/jesslen-od3d/cache/huggingface
export TRANSFORMERS_CACHE=/work/dlclarge1/jesslen-od3d/cache/transformers
echo "Activated meshup_new with CUDA module and env vars."
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
