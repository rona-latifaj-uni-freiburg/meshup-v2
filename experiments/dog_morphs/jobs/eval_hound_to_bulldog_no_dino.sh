#!/bin/bash
#SBATCH --job-name=eval_bulldog_nodino
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/dog_morphs/slurm_logs/eval_bulldog_nodino_%j.out
#SBATCH --error=experiments/dog_morphs/slurm_logs/eval_bulldog_nodino_%j.err

set -euo pipefail
mkdir -p experiments/dog_morphs/slurm_logs
source ./activate_meshup_new.sh

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

OUTPUT_DIR="./experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp"
MESH_PATH="$OUTPUT_DIR/mesh_final/mesh.obj"
TEXT_PROMPT="a bulldog"

if [[ ! -f "$MESH_PATH" ]]; then
  echo "ERROR: Missing mesh at $MESH_PATH"
  exit 1
fi

# Runtime-safe evaluation for 30-minute dev partition limit.
python evaluate_metrics.py \
  --mesh_path "$MESH_PATH" \
  --text_prompt "$TEXT_PROMPT" \
  --output_dir "$OUTPUT_DIR/evaluation" \
  --n_views 8 \
  --n_references 5 \
  --if_model_size XL \
  --if_num_inference_steps 50 \
  --if_guidance_scale 7.5

RESULTS_JSON="$OUTPUT_DIR/evaluation/evaluation_results.json"
if [[ -f "$RESULTS_JSON" ]]; then
  echo "--- Final Metrics Summary ---"
  RESULTS_JSON="$RESULTS_JSON" python - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["RESULTS_JSON"])
data = json.loads(p.read_text())
print(f"FID: {data['fid_score']:.4f}")
print(f"CLIP Similarity: {data['clip_similarity']:.6f}")
PY
else
  echo "ERROR: Missing $RESULTS_JSON"
  exit 1
fi
