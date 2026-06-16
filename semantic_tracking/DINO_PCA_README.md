# DINO PCA Visualization for MeshUp

## Overview

This module provides semantic-based vertex coloring using DINOv2 features and PCA dimensionality reduction. Instead of coloring vertices by their 3D position, it colors them based on semantic features extracted by DINOv2, which better captures correspondence across drastic shape transformations.

## Features

- **Multi-view Feature Extraction**: Renders mesh from multiple viewpoints to extract comprehensive DINO features per vertex
- **PCA-based Coloring**: Reduces high-dimensional DINO features (384D) to RGB colors (3D) using PCA
- **Semantic Correspondence**: Colors reflect semantic similarity rather than geometric position
- **Automatic Integration**: Seamlessly integrates with existing MeshUp tracking pipeline

## Usage

### 1. Enable DINO PCA in Config

Edit your config YAML file (e.g., `configs/your_config.yml`):

```yaml
# Tracking settings
track_correspondence: true
color_method: dino_pca  # Use DINO PCA instead of position-based coloring

# DINO PCA Visualization Settings (optional, these are defaults)
dino_pca_model: dinov2_vits14      # DINO model to use
dino_pca_n_views: 8                # Number of viewpoints for feature extraction
dino_pca_image_size: 512           # Render resolution (higher = slower but more accurate)
```

### 2. Run Training

```bash
python main.py --config configs/your_config.yml
```

### 3. Outputs

The pipeline will generate:

- `colored_meshes/mesh_epoch_*.ply`: Intermediate meshes with DINO PCA colors
- `colored_meshes/mesh_final_correspondence.ply`: Final mesh with DINO PCA colors
- `colored_meshes/mesh_final_position.ply`: Final mesh with position colors (for comparison)
- `colored_meshes/mesh_final_displacement.ply`: Displacement visualization

## Parameters

### `color_method` (string)
- `position`: Color by 3D position (XYZ → RGB) [default]
- `dino_pca`: Color by DINO PCA features (semantic) [new]
- `axis_y`: Color by height (rainbow gradient)
- `cluster`: K-means clustering coloring

### `dino_pca_model` (string)
DINOv2 model variant to use:
- `dinov2_vits14`: ViT-Small (384D features) - **default, fastest**
- `dinov2_vitb14`: ViT-Base (768D features)
- `dinov2_vitl14`: ViT-Large (1024D features)
- `dinov2_vitg14`: ViT-Giant (1536D features) - slowest but most detailed

### `dino_pca_n_views` (int)
Number of viewpoints for feature extraction (default: 8)
- More views = more robust features but slower
- Typical range: 6-12

### `dino_pca_image_size` (int)
Render resolution for feature extraction (default: 512)
- Higher resolution = more detailed features but slower
- Typical range: 256-1024

## Performance Notes

DINO PCA coloring is computed:
1. During training: Every `log_interval_im` epochs
2. At the end: For the final mesh

Typical timing (on GPU):
- 8 views @ 512px: ~5-10 seconds per mesh
- 12 views @ 1024px: ~15-20 seconds per mesh

## Comparison: Position vs DINO PCA

### Position-Based Coloring
- **Pros**: Fast, no extra computation
- **Cons**: Only captures geometric position, not semantic meaning
- **Best for**: Similar shape transformations (e.g., dog → wolf)

### DINO PCA Coloring
- **Pros**: Captures semantic features, better for drastic changes
- **Cons**: Slower (requires rendering + feature extraction)
- **Best for**: Drastic transformations (e.g., quadruped → human, plane → bird)

## Examples

### Test Run
```bash
# Quick test (100 epochs, ~5 minutes)
sbatch jobs/test_dino_pca.sh
```

### Production Run with DINO PCA
```yaml
# configs/hound_to_human_dino_pca.yml
track_correspondence: true
color_method: dino_pca
dino_pca_model: dinov2_vits14
dino_pca_n_views: 8
dino_pca_image_size: 512

output_path: ./outputs/hound_human_dino_pca
mesh: ./meshes/hound.obj
text_prompt: "a human in T-pose"
epochs: 6000
```

## Technical Details

### Algorithm
1. **Multi-view Rendering**: Render mesh from N evenly distributed viewpoints
2. **Feature Extraction**: Extract DINO patch features for each rendered pixel
3. **Vertex Mapping**: Map features back to vertices using barycentric coordinates
4. **Feature Aggregation**: Average features across multiple views
5. **PCA Reduction**: Reduce feature dimensions (384D → 3D)
6. **Normalization**: Normalize to [0, 1] RGB range

### Why PCA?
- DINO features are high-dimensional (384D for ViT-S)
- PCA preserves maximum variance when reducing to 3D
- The 3 principal components map naturally to RGB channels
- Semantically similar regions get similar colors

## Troubleshooting

### "DINO PCA not available" warning
- Check that `semantic_tracking/dino_pca_visualization.py` exists
- Verify DINOv2 is installed: `pip install torch torchvision`

### Out of memory errors
- Reduce `dino_pca_image_size` (e.g., 512 → 256)
- Reduce `dino_pca_n_views` (e.g., 8 → 6)
- Use smaller model (e.g., `dinov2_vits14` instead of larger variants)

### Colors look wrong/random
- DINO PCA colors are semantic, not position-based
- Similar parts should have similar colors
- If colors seem chaotic, the mesh might have issues (check for holes, flipped normals)

## Files

- `semantic_tracking/dino_pca_visualization.py`: Core implementation
- `semantic_tracking/__init__.py`: Module exports
- `loop_tracked.py`: Integration with training loop
- `configs/test_dino_pca.yml`: Example configuration
- `jobs/test_dino_pca.sh`: Example job script

## References

- DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)
- MeshUp: Multi-view Mesh Editing with Score Distillation
- Neural Jacobian Fields for 3D Shape Deformation
