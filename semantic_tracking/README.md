# Semantic Tracking for MeshUp

This module adds semantic correspondence tracking to MeshUp, allowing you to visualize and analyze how mesh parts correspond during deformation.

## Features

1. **Vertex Color Tracking**: Assign colors to vertices that persist through deformation, showing correspondence
2. **Multiple Initialization Methods**: 
   - Position-based (XYZ → RGB)
   - Axis-based (height/depth gradients)
   - Clustering-based (automatic part discovery)
   - Manual part labeling
3. **DINOv2 Feature Loss** (Optional): Add semantic feature consistency during optimization
4. **Export Formats**: PLY with vertex colors, OBJ with texture, JSON correspondence maps

## Quick Start

### 1. Run MeshUp with Tracking

Modify your `main.py` to use the tracked loop:

```python
# In main.py, change:
from loop import loop
# To:
from loop_tracked import loop
```

Then run with the tracking config:

```bash
python main.py --config ./configs/tracked_config.yml \
    --mesh ./meshes/hound.obj \
    --output_path ./outputs/hippo_tracked \
    --text_prompt "a hippo"
```

### 2. View Results

The output directory will contain:
- `colored_meshes/mesh_initial.ply` - Original mesh with colors
- `colored_meshes/mesh_final_correspondence.ply` - Deformed mesh with same colors
- `colored_meshes/mesh_final_displacement.ply` - Colored by displacement magnitude
- `correspondence/final_correspondence.json` - Full correspondence data

### 3. Visualize in MeshLab/Blender

Open both `mesh_initial.ply` and `mesh_final_correspondence.ply`:
- Same-colored vertices show correspondence
- E.g., red vertices in original → red vertices in deformed = same semantic part

## Color Initialization Methods

### Position-Based (Default)
```yaml
color_method: position
```
Maps XYZ coordinates to RGB. Smooth gradient shows spatial correspondence.

### Axis-Based
```yaml
color_method: axis_y  # or axis_x, axis_z
```
Rainbow gradient along one axis. Good for visualizing vertical/horizontal correspondence.

### Clustering
```yaml
color_method: cluster
n_parts: 8
```
K-means clustering into distinct parts. Automatic semantic part discovery.

### Manual Part Labels (Programmatic)
```python
from semantic_tracking import assign_part_colors

tracker = assign_part_colors(
    vertices, faces,
    part_vertices={
        'head': [0, 1, 2, ...],
        'body': [100, 101, ...],
        'left_paw': [200, 201, ...],
    }
)
```

## Adding DINOv2 Feature Loss

Enable in config to encourage semantic consistency:

```yaml
use_dino_loss: true
dino_model: dinov2_vits14  # Smallest/fastest
dino_weight: 0.05
dino_feature_type: cls  # or 'patches' for spatial features
```

This adds a loss that penalizes changes that break DINOv2's semantic understanding.

## API Reference

### VertexColorTracker

```python
from semantic_tracking import VertexColorTracker

# Create tracker
tracker = VertexColorTracker(vertices, faces)

# Initialize colors (choose one method)
tracker.initialize_colors_by_position()
tracker.initialize_colors_by_axis('y', colormap='rainbow')
tracker.initialize_colors_by_clustering(n_parts=8)
tracker.initialize_colors_manual(part_masks, part_colors)

# After deformation
tracker.export_ply("output.ply", vertices=deformed_vertices)
```

### Export Functions

```python
from semantic_tracking import export_mesh_with_colors, export_correspondence_map

# Export colored mesh
export_mesh_with_colors("mesh.ply", vertices, faces, colors, format='ply')

# Export full correspondence data
export_correspondence_map(
    "correspondence.json",
    original_vertices,
    deformed_vertices,
    faces,
    colors=vertex_colors,
    part_labels=part_labels
)
```

## Understanding Correspondence

The key insight is that **vertex indices don't change** during Neural Jacobian Fields deformation - only positions change. By assigning each vertex a color before deformation:

1. **Original mesh**: Vertex 42 at position (x, y, z) has color (r, g, b)
2. **After deformation**: Vertex 42 at position (x', y', z') still has color (r, g, b)

This means:
- Same color = same original vertex = same semantic part
- If cat's right paw vertices are red, and after deformation those red vertices are now in the dog's right paw position, you've established semantic correspondence!

## Examples

Run the demo script:

```bash
python -m semantic_tracking.examples
```

This creates example meshes in `./outputs/` demonstrating all features.
