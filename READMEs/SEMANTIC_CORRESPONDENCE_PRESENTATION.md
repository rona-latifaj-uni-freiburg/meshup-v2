# Bringing Semantic Correspondence to MeshUp

## Presentation Documentation for Text-to-3D Mesh Deformation with Semantic Tracking

**Author:** MeshUp Enhanced Pipeline  
**Date:** 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation & Problem Statement](#2-motivation--problem-statement)
3. [Project Journey & Evolution](#3-project-journey--evolution)
4. [Technical Contributions](#4-technical-contributions)
   - 4.1 [DINOv2 Correspondence Loss](#41-dinov2-correspondence-loss)
   - 4.2 [Cross-Attention Semantic Guidance](#42-cross-attention-semantic-guidance)
   - 4.3 [Vertex Color Tracking](#43-vertex-color-tracking)
   - 4.4 [DINO PCA Visualization](#44-dino-pca-visualization)
   - 4.5 [Target Mesh DINO Guidance](#45-target-mesh-dino-guidance)
   - 4.6 [FID Evaluation](#46-fid-evaluation)
5. [Implementation Details](#5-implementation-details)
6. [Experimental Setup](#6-experimental-setup)
7. [Key Insights & Lessons Learned](#7-key-insights--lessons-learned)
8. [Future Directions](#8-future-directions)

---

## 1. Project Overview

### What is MeshUp?

MeshUp is a text-to-3D mesh deformation framework that transforms a source mesh into a target shape specified by a text prompt. It uses:

- **Neural Jacobian Fields (NJF):** Learns local deformations at each vertex
- **DeepFloyd IF:** Diffusion model for text-conditioned image generation
- **Score Distillation Sampling (SDS):** Optimization via diffusion model gradients

**Original MeshUp Problem:** When deforming meshes, the pipeline had no way to track which parts of the source correspond to which parts of the target.

### What I Added

A complete **semantic correspondence tracking system** that:
- Preserves meaningful part relationships during deformation
- Visualizes semantic consistency using DINOv2 features
- Provides multiple guidance mechanisms for controlled deformation

---

## 2. Motivation & Problem Statement

### The Core Problem

When MeshUp transforms a "dog" mesh into a "dinosaur":
- The dog's head should become the dinosaur's head
- The dog's legs should become the dinosaur's legs
- But without explicit guidance, parts may drift unpredictably

### Why This Matters

- **Animation/Rigging:** Need to transfer bone weights from source to target
- **Shape Interpolation:** Morphing between shapes requires point correspondence
- **Semantic Control:** "Make the head larger" needs to know what the head is

### Research Questions

1. How can we track semantic correspondence during deformation?
2. How can we visualize this correspondence?
3. Can we guide deformation towards specific target configurations?

---

## 3. Project Journey & Evolution

### Approach 1: RAE-DiT (Did Not Work)
**Idea:** Use RAE-DiT (Relation-Aware Encodings with Diffusion Transformers) for semantic correspondence.

**Outcome:** The approach didn't integrate well with MeshUp's pipeline architecture. The diffusion-based approach expected different input formats.

### Approach 2: DINOv2 Correspondence Loss ✅
**Idea:** Use DINOv2 self-supervised features to maintain semantic consistency.

**Key Insight:** DINOv2 learns semantically meaningful features WITHOUT supervision. A dog's head and a dinosaur's head have similar DINO features because they're semantically equivalent.

**Implementation:** Extract DINO features from rendered views, compare to source features, penalize drift.

### Approach 3: Cross-Attention Guidance ✅
**Idea:** The diffusion model's cross-attention already knows "what should be where."

**Key Insight:** When DeepFloyd generates a dinosaur image from text, its attention maps show where each word activates spatially. We can use this as free supervision.

### Approach 4: Vertex Color Tracking ✅
**Idea:** Simple but effective - assign each vertex a unique color based on initial position.

**Key Insight:** NJF maintains vertex indices during deformation. By painting vertices with position-based colors, we can visualize correspondence trivially.

### Approach 5: PCA Visualization ✅
**Idea:** Reproduce the purnasai/Dino_V2 paper's semantic visualization.

**Key Insight:** PCA on DINO features reveals semantic parts. The first PC separates foreground/background; subsequent PCs separate semantic parts (head, body, legs).

---

## 4. Technical Contributions

### 4.1 DINOv2 Correspondence Loss

**File:** `semantic_tracking/dino_correspondence_loss.py`

**Purpose:** Prevent semantic drift during deformation by matching DINO features to source mesh.

#### Architecture

```
Source Mesh (multi-view renders) → DINOv2 → Reference Features
                                              ↓
                                        Store as "anchor"
                                              
During Training:
Current Mesh (multi-view renders) → DINOv2 → Current Features
                                              ↓
                                        Compare to anchor
                                              ↓
                                        L_dino = ||current - reference||²
```

#### Key Components

1. **Global (CLS) Features:** The [CLS] token captures holistic image content
   - Fast to compute
   - Good for overall shape consistency

2. **Spatial (Patch) Features:** Per-patch tokens capture local semantics
   - 16×16 or 37×37 grid of features
   - Enable part-level correspondence
   - Support soft/hard matching

3. **Multi-View Extraction:** Features from 8 canonical viewpoints
   - Ensures all parts are captured
   - Viewpoint-consistent comparison

#### Hyperparameters

```yaml
dino_weight: 0.5          # Overall DINO loss weight
dino_global_weight: 0.3   # CLS token weight
dino_spatial_weight: 0.7  # Patch tokens weight
dino_warmup_epochs: 50    # Epochs before full activation
dino_n_views: 8           # Number of viewpoints
```

#### Key Equations

**Global (CLS) Loss:**
$$L_{global} = 1 - \frac{1}{N} \sum_{i=1}^{N} cos(c_i^{ref}, c_i^{curr})$$

**Spatial (Patch) Loss with Soft Matching:**
$$L_{spatial} = -\frac{1}{N} \sum_{i} \log \frac{\exp(sim(f_i^{curr}, f_i^{ref})/\tau)}{\sum_j \exp(sim(f_i^{curr}, f_j^{ref})/\tau)}$$

---

### 4.2 Cross-Attention Semantic Guidance

**File:** `semantic_tracking/cross_attention_guidance.py`

**Purpose:** Use diffusion model's internal attention maps for semantic guidance.

#### Key Insight

When DeepFloyd generates "a dinosaur" from text:
- The word "head" activates in the head region
- The word "tail" activates in the tail region
- These attention maps are FREE semantic supervision!

#### Architecture

```
Rendered Image + Text Prompt → DeepFloyd UNet → Cross-Attention Maps
                                                      ↓
                                               Extract attention
                                                      ↓
                                          Compare to reference attention
                                                      ↓
                                          L_attn = consistency + entropy + coverage
```

#### Loss Components

1. **Consistency Loss:** Attention patterns should match reference
2. **Entropy Loss:** Attention should be focused, not diffuse
3. **Coverage Loss:** Ensure all parts receive attention

#### Configuration

```yaml
cross_attn_weight: 0.3            # Overall weight
cross_attn_consistency_weight: 0.5
cross_attn_entropy_weight: 0.2
cross_attn_coverage_weight: 0.3
cross_attn_layers: [5, 6, 7]      # Which UNet layers to use
cross_attn_token_idx: [2, 3, 4]   # Which text tokens to track
```

---

### 4.3 Vertex Color Tracking

**File:** `semantic_tracking/vertex_color_tracking.py`

**Purpose:** Simple, elegant correspondence visualization via vertex colors.

#### Key Insight

NJF deforms meshes by predicting per-vertex Jacobians. **Vertex indices remain constant!**

This means: If vertex 42 starts at the dog's nose, vertex 42 will still be at the corresponding point on the dinosaur.

#### Color Assignment Methods

```python
# 1. Position-based: RGB = normalized (x, y, z)
color = (vertex_pos - min_pos) / (max_pos - min_pos)

# 2. Axis-based: Gradient along one axis
color = colormap(vertex_pos[:, axis])

# 3. Clustering: K-means on positions
cluster_labels = kmeans.fit_predict(vertex_pos)
color = cluster_colors[cluster_labels]
```

#### Usage

```python
tracker = VertexColorTracker(n_vertices=mesh.v_pos.shape[0])
tracker.initialize_colors_by_position(initial_positions)

# During training - no update needed!
# Vertex indices stay constant

# Export colored mesh
export_mesh_with_colors(mesh, tracker.colors, "output.ply")
```

---

### 4.4 DINO PCA Visualization

**File:** `semantic_tracking/dino_pca_visualization.py` and `loop_tracked.py::save_combined_pca_visualization()`

**Purpose:** Produce semantically meaningful color visualizations matching the purnasai/Dino_V2 paper.

#### The 2-Step PCA Algorithm

**Step 1: Foreground/Background Separation**
```python
# Extract DINO features from all views
features = dino_model(all_images)  # (n_views * H * W, D)

# First PCA on all features
pca1 = PCA(n_components=3)
pc1 = pca1.fit_transform(features)[:, 0]

# PC1 separates fg/bg - threshold at 0.35
pc1_scaled = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
bg_mask = pc1_scaled > 0.35  # Background pixels
fg_mask = ~bg_mask           # Foreground pixels
```

**Step 2: Semantic Coloring on Foreground**
```python
# Second PCA on foreground only
pca2 = PCA(n_components=3)
fg_pca = pca2.fit_transform(features[fg_mask])

# Min-max scale for RGB
for i in range(3):
    fg_pca[:, i] = (fg_pca[:, i] - fg_pca[:, i].min()) / 
                   (fg_pca[:, i].max() - fg_pca[:, i].min())

# Final colors: black background, PCA foreground
colors = np.zeros((len(features), 3))
colors[fg_mask] = fg_pca
```

#### Critical Implementation Details

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | `dinov2_vitl14` | Large model (1024D) captures more semantics |
| Normalization | mean=0.5, std=0.2 | Simple normalization (NOT ImageNet stats) |
| Threshold | 0.35 (fixed) | Paper's threshold for fg/bg |
| Mask Logic | bg > 0.35, fg ≤ 0.35 | Background has HIGHER PC1 values |

#### Why Combined PCA Across Views?

**Problem with Per-View PCA:**
```
View 1: PC1 = position.x → left wing = RED, right wing = BLUE
View 2: PC1 = intensity  → left wing = GREEN, right wing = GREEN
```
Different PCs capture different variance → inconsistent colors!

**Solution - Combined PCA:**
```
All Views: Fit single PCA → consistent semantic mapping
View 1: left wing = RED, right wing = RED (same semantic part!)
View 2: left wing = RED, right wing = RED
```

---

### 4.5 Target Mesh DINO Guidance

**Files:** 
- `semantic_tracking/target_mesh_dino_guidance.py` (V1)
- `semantic_tracking/target_mesh_dino_guidance_v2.py` (V2 - viewpoint-aligned)

**Purpose:** Guide deformation towards a specific target mesh, not just text.

#### Use Case

Instead of "dog → text prompt → something dinosaur-like":
```
dog.obj → dinosaur_014.obj (specific mesh!)
```

#### V1 vs V2 Comparison

**V1 (Simple):**
- Extract target features from N fixed viewpoints
- Compare source render to nearest target feature
- Problem: Misaligned viewpoints cause bad matching

**V2 (Viewpoint-Aligned):**
- Pre-compute target features on dense grid (e.g., 12 azimuths × 3 elevations)
- For each source render, find CLOSEST target viewpoint
- Compare only aligned views
- Much better spatial correspondence!

#### Configuration

```yaml
target_mesh_path: "./meshes/dinosaur_014.obj"
target_mesh_weight: 0.5
target_mesh_warmup: 50
target_mesh_global_weight: 0.3
target_mesh_spatial_weight: 0.7
```

---

### 4.6 FID Evaluation

**Files:**
- `semantic_tracking/fid_evaluation.py` (Core module)
- `evaluate_fid.py` (Standalone evaluation script)
- `semantic_tracking/multiangle_visualization.py` (Multi-angle rendering)

**Purpose:** Provide quantitative evaluation of mesh-to-text alignment via Fréchet Inception Distance.

#### Motivation

Visual inspection alone is subjective. FID provides a **numerical metric** to:
1. Compare different methods objectively
2. Track improvement across training runs
3. Present quantitative results in papers/presentations

#### Architecture

```
Text Prompt ──────────────────────────────────────────────────────┐
      │                                                            │
      ▼                                                            │
 DeepFloyd IF XL                                                   │
      │                                                            │
      ▼                                                            │
Reference Images ══════════╗                                       │
(N generated, H×W)         ║                                       │
                           ║                                       │
      ┌────────────────────╬──────────────────────────────────────┤
      │                    ║                                       │
      │                    ▼                                       │
      │       ┌────────────────────────┐                           │
      │       │    Inception V3        │ ◄─────────────────────────┘
      │       │   (pretrained)         │
      │       └────────────────────────┘
      │                    │
      │                    ▼
      │         Pool3 Features (2048-D)
      │                    │
      │          ┌─────────┴─────────┐
      │          │                   │
      │          ▼                   ▼
      │    μ_ref, Σ_ref        μ_mesh, Σ_mesh
      │          │                   │
      │          └─────────┬─────────┘
      │                    │
      │                    ▼
      │    ┌─────────────────────────────────┐
      │    │   FID = ||μ_ref - μ_mesh||²     │
      │    │        + Tr(Σ_ref + Σ_mesh      │
      │    │           - 2(Σ_ref·Σ_mesh)^½)  │
      │    └─────────────────────────────────┘
      │                    │
      │                    ▼
      │               FID Score
      │             (lower = better)
      │
Mesh ─┘
  │
  ▼
8 Viewpoint Renders ══════════════════════════════════════════════╝
(fixed angles: 0°, 45°, 90°, ..., 315° azimuth)
```

#### Key Components

**1. InceptionV3 Feature Extractor:**
```python
class InceptionV3Features(nn.Module):
    def __init__(self):
        # Load pretrained Inception V3
        # Truncate at pool3 layer → 2048-D features
        # Normalize with ImageNet stats
```

**2. FID Computation:**
```python
def compute_fid(real_features, generated_features):
    # Compute mean and covariance for each set
    mu_real, sigma_real = mean(real), cov(real)
    mu_gen, sigma_gen = mean(gen), cov(gen)
    
    # Fréchet distance formula
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)
    
    fid = diff @ diff + trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
```

**3. Multi-Angle Mesh Rendering:**
```python
class MultiAnglePCAVisualizer:
    ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 fixed viewpoints
    
    def render_all_angles(self, mesh, renderer, resolution=512):
        renders = []
        for azimuth in self.ANGLES:
            image = render_at_angle(mesh, azimuth, elevation=0)
            renders.append(image)
        return renders
```

#### Usage

**Standalone Evaluation:**
```bash
python evaluate_fid.py \
    --mesh_path outputs/hound_to_hippo/mesh_final/mesh_final.obj \
    --text_prompt "a hippo, realistic" \
    --output_dir outputs/hound_to_hippo/evaluation \
    --n_reference 50
```

**In Training Loop:**
```python
from semantic_tracking import FIDEvaluator, compute_mesh_fid

# After training completes
fid_score = compute_mesh_fid(
    mesh=final_mesh,
    text_prompt=config["text_prompt"],
    n_reference_images=50,
    render_resolution=512
)
print(f"Final FID: {fid_score:.2f}")
```

#### Output Structure

```
evaluation/
├── mesh_renders/           # 8 viewpoint renders
│   ├── angle_000.png
│   ├── angle_045.png
│   └── ...
├── reference_images/       # Generated from text
│   ├── ref_0000.png
│   └── ...
└── fid_results.json        # {"fid_score": 45.23, "n_reference": 50, ...}
```

#### Interpretation

| FID Range | Quality |
|-----------|---------|
| < 30 | Excellent alignment |
| 30-60 | Good alignment |
| 60-100 | Moderate alignment |
| > 100 | Poor alignment |

**Note:** FID is relative - compare across your own experiments rather than to absolute thresholds.

---

## 5. Implementation Details

### Module Structure

```
semantic_tracking/
├── __init__.py
├── dino_correspondence_loss.py      # Core DINO loss
├── cross_attention_guidance.py      # Diffusion attention guidance
├── vertex_color_tracking.py         # Color-based tracking
├── dino_pca_visualization.py        # PCA colorization for 3D
├── target_mesh_dino_guidance.py     # Target mesh guidance V1
├── target_mesh_dino_guidance_v2.py  # Target mesh guidance V2
├── fid_evaluation.py                # FID metric computation
├── multiangle_visualization.py      # Multi-angle PCA rendering
└── correspondence_export.py         # Export utilities

evaluate_fid.py                      # Standalone FID evaluation script
```

### Integration Points in `loop_tracked.py`

1. **Imports** (lines 55-115): Conditional imports with fallback
2. **TrackedVisualizer** (line 123): Extended visualization class
3. **save_combined_pca_visualization** (line 276): 220+ lines PCA method
4. **Training loop**: DINO loss computation + visualization calls

### Config Options

```yaml
# Enable/disable components
use_dino_correspondence: true
use_cross_attention_guidance: false
track_correspondence: true
save_pca_visualization: true

# DINO Loss
dino_weight: 0.5
dino_global_weight: 0.3
dino_spatial_weight: 0.7
dino_warmup_epochs: 50
dino_n_views: 8
dino_model_name: "dinov2_vits14_reg"

# PCA Visualization
pca_interval: 150
pca_render_res: 518

# Tracking
color_method: "position"  # "position", "axis", "clustering"
```

---

## 6. Experimental Setup

### Job Script Example

**File:** `dino_jobs/job_h100_hound_human_dino_pca.sh`

```bash
#!/bin/bash
#SBATCH --job-name=hound_human_dino
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

python main.py \
    --config ./configs/hound_human_dino_simple.yml \
    --mesh ./meshes/Omni6DPose/OmniObject3D/hound.obj \
    --output_path ./outputs/hound_to_human_dino \
    --text_prompt "a human, realistic proportions, standing pose" \
    --track_correspondence \
    --use_dino_loss \
    --save_pca
```

### Typical Config

**File:** `configs/hound_human_dino_simple.yml`

```yaml
name: "hound_to_human_dino"
train_res: 256
n_views: 4
n_epochs: 1000
lr: 5e-4

# DINO Configuration
use_dino_correspondence: true
dino_weight: 0.5
dino_warmup_epochs: 50

# PCA Visualization
save_pca_visualization: true
pca_interval: 100

# Tracking
track_correspondence: true
color_method: "position"
```

---

## 7. Key Insights & Lessons Learned

### Insight 1: Self-Supervised Features Are Powerful

DINOv2 learns semantic features without any labels. A dog's head and a cat's head have similar features because they're semantically equivalent "heads."

**Implication:** We get semantic correspondence "for free" from pretrained models.

### Insight 2: Combined PCA is Essential

Per-image PCA produces inconsistent colors because different views have different dominant variance directions.

**Solution:** Fit PCA across ALL views jointly, then apply to each.

### Insight 3: Warmup Prevents Collapse

Applying strong DINO constraints from epoch 0 prevents the mesh from deforming at all.

**Solution:** Linear warmup over 50-100 epochs lets the mesh establish basic shape before correspondence constraints activate.

### Insight 4: Cross-Attention is Free Supervision

The diffusion model already knows semantic layout. Its attention maps tell us "where should the head be?" without any extra computation.

### Insight 5: Vertex Indices are Your Friend

NJF maintains vertex correspondence by design. No need for complex matching - just track colors by index.

---

## 8. Future Directions

### Near-Term

1. **Ablation Studies:** Quantify contribution of each component
2. **Dense Correspondence:** Per-vertex feature instead of per-view
3. **Real-Time Visualization:** Interactive PCA during training

### Long-Term

1. **Correspondence Transfer:** Use tracked correspondence for rigging transfer
2. **Part-Aware Deformation:** Explicit part segmentation + per-part control
3. **Multi-Target Blending:** Combine features from multiple targets

---

## Appendix: Code References

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| [dino_correspondence_loss.py](semantic_tracking/dino_correspondence_loss.py) | DINO feature consistency | `DINOv2Extractor`, `DINOCorrespondenceLoss` |
| [cross_attention_guidance.py](semantic_tracking/cross_attention_guidance.py) | Diffusion attention guidance | `CrossAttentionGuidance` |
| [vertex_color_tracking.py](semantic_tracking/vertex_color_tracking.py) | Color-based tracking | `VertexColorTracker` |
| [dino_pca_visualization.py](semantic_tracking/dino_pca_visualization.py) | 3D PCA coloring | `DINOPCAColorizer` |
| [target_mesh_dino_guidance.py](semantic_tracking/target_mesh_dino_guidance.py) | Target mesh guidance | `TargetMeshDINOGuidance` |
| [loop_tracked.py](loop_tracked.py) | Main training loop | `TrackedVisualizer`, `save_combined_pca_visualization` |

---

## Quick Reference: Running Experiments

```bash
# 1. Activate environment
source activate_meshup_new.sh

# 2. Simple DINO-only experiment
python main.py \
    --config ./configs/hound_human_dino_simple.yml \
    --mesh ./meshes/hound.obj \
    --output_path ./outputs/hound_human \
    --text_prompt "a human"

# 3. Submit H100 job
sbatch dino_jobs/job_h100_hound_human_dino_pca.sh

# 4. Monitor training
tail -f slurm_logs/hound_human_dino_*.out

# 5. View results
ls outputs/hound_human/pca_visualization/
```

---

*End of Documentation*
