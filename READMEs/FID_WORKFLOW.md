# FID Evaluation Workflow

## Overview

The FID (Fréchet Inception Distance) evaluation workflow consists of two steps:

1. **Training Phase**: Run the mesh deformation training with DINO loss for semantic correspondence
2. **Evaluation Phase**: Compute FID scores comparing mesh renders to reference images

---

## Complete Workflow

### Option 1: Run Everything Together (Recommended)

Submit a job that handles both training and evaluation:

```bash
# HIPPO
sbatch jobs/hound_to_hippo_with_fid.sh

# HUMAN
sbatch jobs/hound_to_human_with_fid.sh
```

**Time:** ~30-60 minutes (depending on GPU availability)

**Output:**
```
outputs/hound_to_hippo_fid/
├── evaluation/
│   ├── mesh_renders/           # 8 mesh views
│   │   ├── view_00.png
│   │   └── ...
│   ├── reference_images/       # 16 generated reference images
│   │   ├── ref_00.png
│   │   └── ...
│   └── fid_results.json        # FID scores
├── [training outputs...]
```

### Option 2: Manual Two-Step Process

#### Step 1: Train the mesh

```bash
python main.py --config ./configs/tracked_config.yml \
  --mesh ./meshes/hound.obj \
  --output_path ./outputs/my_experiment \
  --text_prompt "a hippo" \
  --color_method dino_pca \
  --use_dino_loss \
  --dino_weight 0.08 \
  --regularize_jacobians_weight 35000 \
  --epochs 3000 \
  --log_interval_im 150
```

#### Step 2: Evaluate FID

```bash
python evaluate_fid.py \
  --mesh_path ./outputs/my_experiment/mesh_final/mesh.obj \
  --text_prompt "a hippo" \
  --output_dir ./outputs/my_experiment/evaluation \
  --n_views 8 \
  --n_references 16
```

---

## Reading FID Results

After evaluation, check the results JSON:

```bash
cat outputs/hound_to_hippo_fid/evaluation/fid_results.json
```

Example output:
```json
{
  "fid_score": 42.15,
  "mesh_path": "outputs/hound_to_hippo_fid/mesh_final/mesh.obj",
  "text_prompt": "a hippo",
  "n_views": 8,
  "n_references": 16
}
```

### Interpreting FID Scores

| FID Range | Quality |
|-----------|---------|
| < 30 | Excellent - Strong alignment between mesh and text |
| 30-60 | Good - Reasonable alignment |
| 60-100 | Moderate - Some alignment but room for improvement |
| > 100 | Poor - Weak alignment |

**Note:** FID is most useful for comparing your own experiments. Absolute values vary by task.

---

## What Gets Computed

### 1. Mesh Renders (8 viewpoints)
- Fixed azimuth angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
- Resolution: 512×512
- Format: PNG
- Camera: Fixed elevation=0°, distance=3.0, FOV=60°

### 2. Reference Images (16 images)
- Generated using DeepFloyd IF-XL diffusion model
- Text prompt: The `--text_prompt` you specified
- Resolution: 512×512 (native DeepFloyd output)
- Format: PNG
- Seed: Varies from 42 to 57 for diversity

### 3. FID Computation
- Uses Inception-v3 pool3 features (2048-dimensional)
- Compares feature distributions:
  - Mean: μ_mesh vs μ_ref
  - Covariance: Σ_mesh vs Σ_ref
- Fréchet distance formula: 
  ```
  FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
  ```

---

## Key Configuration Details

### Job Script

- **Partition:** `dev_gpu_h100` (single H100 GPU)
- **Time:** 60 minutes
- **Memory:** 40GB
- **CPUs:** 4

### Training Parameters

```yaml
epochs: 3000
dino_weight: 0.08              # DINO loss strength
regularize_jacobians_weight: 35000
log_interval_im: 150           # PCA viz every 150 epochs
color_method: dino_pca         # Semantic coloring
```

### Evaluation Parameters

```yaml
n_views: 8                  # 8 fixed viewpoints for mesh renders
n_references: 16            # 16 reference images from DeepFloyd
guidance_scale: 7.5         # DeepFloyd guidance intensity
num_inference_steps: 50     # Diffusion steps
```

---

## Troubleshooting

### "Final mesh not found"

**Problem:** Job can't find `mesh_final/mesh.obj`  
**Solution:** Check the training completed successfully
```bash
tail -50 outputs/hound_to_hippo_fid/logs/main.log
```

### Out of memory errors

**Problem:** GPU runs out of memory during FID computation  
**Solution:** Reduce `--n_references` (e.g., 8 instead of 16)

### DeepFloyd model download fails

**Problem:** First-time DeepFloyd FXL download fails  
**Solution:** The model is ~20GB, needs good internet connection. Check free space with:
```bash
df -h
```

---

## Comparing Multiple Experiments

To compare results across different settings, create a summary:

```bash
for exp in outputs/*/evaluation/fid_results.json; do
  dirname=$(dirname $(dirname $exp))
  echo "$(basename $dirname): $(grep fid_score $exp | cut -d: -f2)"
done
```

---

## Next Steps

After getting FID scores:

1. **Analyze results** - Compare different hyperparameters
2. **Ablation studies** - Test without DINO loss: `--use_dino_loss false`
3. **Different meshes** - Try other source meshes or targets
4. **Visualize renders** - View mesh renders and reference images side-by-side

