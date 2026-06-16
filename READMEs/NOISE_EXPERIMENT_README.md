# Noise Sensitivity Experiment

## Overview
This experiment tests how different random noise initialization affects mesh deformation outcomes in Score Distillation Sampling (SDS).

## Research Question
**Does different noise conditioning lead to significantly different deformation results, or is the process relatively stable?**

## Experimental Design

### Two Transformations Tested:

#### 1. **Plane → Bird** (Good Semantic Correspondence)
- **Mesh**: Toy airplane
- **Target**: "an eagle"
- **Seeds**: 1111, 2222, 3333
- **Why**: Planes and birds have natural correspondence (wings→wings, fuselage→body, tail→tail)
- **Expected**: LOWER variance - natural correspondence should guide deformation consistently
- **Runtime**: ~1 hour each (4000 epochs)

#### 2. **Hound → Human** (Challenging Correspondence)
- **Mesh**: Dog (quadruped)
- **Target**: "a human person standing upright"
- **Seeds**: 5555, 6666, 7777
- **Why**: Quadruped to biped is challenging (front paws→hands, 4 legs→2 legs, snout→nose)
- **Expected**: HIGHER variance - ambiguous correspondence may lead to diverse solutions
- **Runtime**: ~1.5 hours each (5000 epochs)

## Running the Experiment

```bash
# Submit all 6 jobs at once
bash jobs/RUN_ALL_NOISE_TESTS.sh

# Or submit individually
sbatch jobs/noise_plane_bird_seed1.sh
sbatch jobs/noise_plane_bird_seed2.sh
# ... etc

# Monitor progress
squeue --me
```

## Output Structure

Each job produces:
```
outputs/noise_test/{transformation}_seed{N}/
├── colored_meshes/
│   ├── mesh_final_correspondence.ply  # DINO PCA colors (semantic)
│   ├── mesh_final_position.ply        # Position-based colors (baseline)
│   ├── mesh_final_displacement.ply    # Deformation heatmap
│   ├── mesh_epoch_200.ply             # Intermediate snapshots
│   ├── mesh_epoch_400.ply
│   └── ...
├── correspondence/
│   └── final_correspondence.json      # Vertex displacement data
├── mesh_final/
│   └── mesh.obj                       # Final textured mesh
└── images/
    └── epoch_*.png                    # Training renders
```

## Analysis: What to Look For

### 1. **Visual Inspection** (Load all 3 seeds in MeshLab/Blender)
   - **DINO PCA colors**: Do similar parts get similar colors across seeds?
   - **Shape**: Are the overall shapes similar or drastically different?
   - **Correspondence**: Does paw→hand mapping stay consistent?

### 2. **Quantitative Metrics**

#### A. Cross-Seed Correspondence Quality
For each transformation, compare the 3 seeds:
```python
# Load all 3 final meshes
plane_bird_seeds = [load_mesh(f"outputs/noise_test/plane_bird_seed{s}/colored_meshes/mesh_final_correspondence.ply") 
                    for s in [1111, 2222, 3333]]

# Compare DINO PCA colors (should be similar if correspondence is stable)
color_variance = np.var([m.vertex_colors for m in plane_bird_seeds], axis=0).mean()
print(f"Color variance across seeds: {color_variance:.4f}")
# LOW variance → stable correspondence
# HIGH variance → noise-dependent correspondence
```

#### B. Vertex Displacement Consistency
```python
# Load displacement data
import json
disps = []
for seed in [1111, 2222, 3333]:
    with open(f"outputs/noise_test/plane_bird_seed{seed}/correspondence/final_correspondence.json") as f:
        data = json.load(f)
        original = np.array(data['original_vertices'])
        deformed = np.array(data['deformed_vertices'])
        disps.append(deformed - original)

# Compute displacement field similarity
mean_disp_diff = np.mean([np.linalg.norm(disps[i] - disps[j], axis=1).mean() 
                          for i in range(3) for j in range(i+1, 3)])
print(f"Mean displacement difference: {mean_disp_diff:.4f}")
# LOW → consistent deformation across seeds
# HIGH → noise-sensitive deformation
```

#### C. Shape Similarity (Chamfer Distance)
```python
from scipy.spatial.distance import cdist

def chamfer_distance(mesh1, mesh2):
    d1 = cdist(mesh1.vertices, mesh2.vertices).min(axis=1).mean()
    d2 = cdist(mesh2.vertices, mesh1.vertices).min(axis=1).mean()
    return (d1 + d2) / 2

# Compare all pairs
for i, j in [(0,1), (0,2), (1,2)]:
    cd = chamfer_distance(plane_bird_seeds[i], plane_bird_seeds[j])
    print(f"Chamfer distance seed{i+1} vs seed{j+1}: {cd:.4f}")
```

## Expected Results

### Plane → Bird (Good Correspondence)
**Hypothesis**: LOW noise sensitivity
- DINO PCA colors should be very similar across seeds
- Wing positions should be consistent
- Tail orientation should be similar
- **Why**: Strong semantic priors from DINO + natural correspondence guides the deformation

### Hound → Human (Challenging Correspondence)
**Hypothesis**: MODERATE to HIGH noise sensitivity
- Color patterns may vary (different part mappings)
- Pose might differ (standing vs. slightly bent)
- Arm/leg positions could vary
- **Why**: Ambiguous 4→2 limb mapping + quadruped→biped challenge allows noise to influence outcome

## Interpretation Guide

| Observation | Interpretation |
|------------|---------------|
| **Colors very similar across seeds** | Stable semantic correspondence - DINO guidance is strong |
| **Colors quite different** | Noise affects which source parts map to which target parts |
| **Shapes nearly identical** | SDS is deterministic given similar correspondence |
| **Shapes vary significantly** | Either: (1) correspondence differs OR (2) SDS optimization is chaotic |
| **Low variance for plane, high for hound** | Confirms: easy tasks are stable, hard tasks are noise-sensitive |
| **High variance for both** | Suggests: SDS with this guidance is generally non-deterministic |
| **Low variance for both** | Suggests: Strong regularization + DINO make process very stable |

## PCA Explained Variance

Watch the PCA explained variance in the logs:
```
PCA explained variance: [0.437, 0.239, 0.195]
```

- Sum of first 3 components ~87% → Good semantic structure captured
- If variance ratios differ significantly across seeds → Different semantic interpretations
- If variance ratios are similar → Consistent semantic feature distribution

## Comparison Workflow

1. **Visually compare in MeshLab**:
   ```bash
   meshlab outputs/noise_test/plane_bird_seed1111/colored_meshes/mesh_final_correspondence.ply \
           outputs/noise_test/plane_bird_seed2222/colored_meshes/mesh_final_correspondence.ply \
           outputs/noise_test/plane_bird_seed3333/colored_meshes/mesh_final_correspondence.ply
   ```

2. **Load displacement heatmaps**:
   - Purple = no movement
   - Yellow/Green = maximum movement
   - Check if deformation patterns are similar

3. **Animate the sequence** (optional):
   Load `mesh_epoch_*.ply` files in sequence to create animation of transformation

## Computational Cost

- **Total GPU hours**: ~7.5 hours (3×1h + 3×1.5h)
- **Parallelizable**: Yes, all 6 jobs can run simultaneously
- **Memory**: 64GB per job
- **Storage**: ~2GB per job × 6 = 12GB total

## Follow-up Questions

If results show HIGH variance:
- Does increasing DINO weight reduce variance?
- Does Cross-Attention guidance help stabilize?
- Is variance in correspondence or just final pose?

If results show LOW variance:
- Test with NO DINO/NO guidance (pure SDS) - does variance increase?
- Test with much longer runs - do solutions converge?
- Test with extreme transformations (train→bird) - still stable?
