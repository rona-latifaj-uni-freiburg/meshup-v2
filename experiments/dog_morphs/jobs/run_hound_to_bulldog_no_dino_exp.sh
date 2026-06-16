#!/bin/bash
#SBATCH --job-name=hound_bulldog_nodino_exp
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/dog_morphs/slurm_logs/hound_bulldog_nodino_exp_%j.out
#SBATCH --error=experiments/dog_morphs/slurm_logs/hound_bulldog_nodino_exp_%j.err

set -euo pipefail
mkdir -p experiments/dog_morphs/slurm_logs
source ./activate_meshup_new.sh

export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
mkdir -p "$TRANSFORMERS_CACHE"

CONFIG_FILE="./experiments/dog_morphs/configs/hound_to_bulldog_no_dino_config.yml"
OUTPUT_DIR=$(grep 'output_path:' "$CONFIG_FILE" | awk '{print $2}')
TEXT_PROMPT=$(grep 'text_prompt:' "$CONFIG_FILE" | cut -d '"' -f 2)

# 1) Run mesh deformation training
echo "--- Starting Mesh Deformation (Hound -> Bulldog, NO DINO loss) ---"
python main.py --config "$CONFIG_FILE"
echo "--- Mesh Deformation Finished ---"

# 2) Generate PCA evolution visualization
echo "--- Generating PCA Evolution Visualization ---"
python generate_pca_evolution_4views.py \
    --epoch_renders_dir "$OUTPUT_DIR/epoch_renders" \
    --output_dir "$OUTPUT_DIR/pca_evolution" \
    --epochs "1,750,1500,2250,3000" \
    --crop_margin_ratio 0.08 \
    --min_crop_side 0
echo "--- PCA Evolution Visualization Finished ---"

# 3) Evaluate final mesh with FID + CLIP
echo "--- Running Evaluation (FID and CLIP) ---"
MESH_PATH="$OUTPUT_DIR/mesh_final/mesh.obj"
if [[ -f "$MESH_PATH" ]]; then
    if ! python evaluate_metrics.py \
        --mesh_path "$MESH_PATH" \
        --text_prompt "$TEXT_PROMPT" \
        --output_dir "$OUTPUT_DIR/evaluation" \
        --n_views 8 \
        --n_references 16 \
        --if_model_size XL \
        --if_num_inference_steps 50 \
        --if_guidance_scale 7.5; then
        echo "WARN: HQ evaluation failed, retrying with low-memory settings"
        python evaluate_metrics.py \
            --mesh_path "$MESH_PATH" \
            --text_prompt "$TEXT_PROMPT" \
            --output_dir "$OUTPUT_DIR/evaluation" \
            --n_views 8 \
            --n_references 8 \
            --if_model_size M \
            --if_num_inference_steps 20 \
            --if_guidance_scale 5.0 \
            --if_height 256 \
            --if_width 256 \
            --if_cpu_offload
    fi
else
    echo "ERROR: Final mesh not found at $MESH_PATH. Skipping evaluation."
fi

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
    echo "ERROR: Missing evaluation results JSON at $RESULTS_JSON"
    exit 1
fi

echo "--- Evaluation Finished ---"

# 4) Keep output clean
echo "--- Cleaning Non-Essential Outputs ---"
rm -f "$OUTPUT_DIR/colored_meshes/mesh_final_correspondence.ply"
rm -f "$OUTPUT_DIR/colored_meshes/mesh_final_dino_pca.ply"
rm -f "$OUTPUT_DIR/colored_meshes/mesh_final_displacement.ply"

rm -rf "$OUTPUT_DIR/figure"
rm -rf "$OUTPUT_DIR/grads"
rm -rf "$OUTPUT_DIR/epoch_renders"
rm -rf "$OUTPUT_DIR/mesh_best_clip"
rm -rf "$OUTPUT_DIR/mesh_best_total"
rm -rf "$OUTPUT_DIR/mesh_final_texture"
rm -rf "$OUTPUT_DIR/mesh_mesh_log"
rm -rf "$OUTPUT_DIR/n_vert"
rm -rf "$OUTPUT_DIR/pca_evolution_4views"

echo "--- Cleanup Finished ---"
