# Scientific Presentation Outline: Current Mesh Deformation Pipeline

Date: 2026-07-22

## One-Sentence Pitch

This project turns cross-animal deformation into a constrained 3D optimization problem: a source mesh is deformed toward a target mesh using semantically aligned local geometry losses, while Jacobian and edge regularizers prevent spikes, tearing, and locally incoherent motion.

## Core Thesis For The Presentation

The important move is that the pipeline is no longer asking a 2D image model to guess what the 3D object should become. It uses the target mesh directly.

The current method says:

> Match the target geometry, but only compare compatible semantic regions, and keep the deformation field locally smooth.

That is the scientific story:

1. Global geometry gives coarse shape alignment.
2. Semantic buckets give part-aware correspondences.
3. Jacobian regularization makes the deformation spatially coherent.
4. Edge displacement guards suppress spikes and folding artifacts.
5. DenseCorr3D gives the strongest quantitative setting because the labels come from supervised annotated meshes.

## Recommended Slide Outline

### Slide 1: Title

**Part-Aware 3D Mesh Deformation With Semantic Chamfer Guidance**

Subtitle:

**From image-prior MeshUp to geometry-first, supervised-label deformation**

What to say:

> I started from the MeshUp image-guided pipeline, but the current pipeline is now a direct 3D optimization system. The deformation is driven by local semantic geometry rather than by sampled 2D diffusion images.

### Slide 2: Problem Formulation

Inputs:

- Source mesh: `M_s = (V_s, F_s)`
- Target mesh: `M_t = (V_t, F_t)`
- Source semantic labels: `l_s(i) in {1, ..., K}`
- Target semantic labels: `l_t(j) in {1, ..., K}`

Goal:

Find deformed vertices `V(theta)` such that:

1. The deformed source matches the target shape.
2. Source part `k` matches target part `k`.
3. The deformation remains smooth and physically plausible.

Mathematically:

```text
theta* = argmin_theta L_total(theta)
```

where `theta` parameterizes the deformation of the source mesh.

What to say:

> The key is that I am not optimizing independent vertex positions. I optimize a deformation field, and I supervise it using target geometry plus semantic labels.

### Slide 3: Pipeline Overview

Current pipeline:

```text
source mesh
  -> mesh normalization
  -> semantic bucket construction
  -> initialize identity deformation
  -> 4000 epoch geometry optimization
  -> final deformed mesh
  -> global, part-level, and distortion evaluation
```

For the strongest current runs:

- Optimization length: `4000` epochs
- Parameterization: Jacobian deformation
- Main loss: hard PartField/DenseCorr3D bucket Chamfer
- Global target Chamfer: implemented, but disabled in the best pf-only runs
- Image losses: disabled
- SDS: disabled
- DINO image guidance: disabled

What to say:

> The pipeline became simpler and stronger: it uses explicit target meshes and explicit local correspondences instead of optimizing through a stochastic image generation prior.

### Slide 4: Full Objective Function

General objective supported by the code:

```text
L_total(theta)
  = lambda_pf      L_partfield(theta)
  + lambda_global  L_global_chamfer(theta)
  + lambda_jn      L_jacobian_neighbor(theta)
  + lambda_jump    L_edge_displacement_jump(theta)
  + lambda_jac     L_identity_jacobian(theta)
  + lambda_out     L_jacobian_outlier(theta)
  + lambda_edge    L_edge_stretch(theta)
  + lambda_img     L_image(theta)
```

For the final strongest hard bucket runs:

```text
lambda_pf     = 8000
lambda_jn     = 1000
lambda_jump   = 500
lambda_global = 0
lambda_jac    = 0
lambda_out    = 0
lambda_edge   = 0
lambda_img    = 0
```

So the active final objective is:

```text
L_total(theta)
  = 8000 L_partfield(theta)
  + 1000 L_jacobian_neighbor(theta)
  + 500  L_edge_displacement_jump(theta)
```

What to say:

> The important point is that the code supports a large family of losses, but the best current recipe is compact and interpretable. It is part-aware geometry matching plus two deformation-stability terms.

### Slide 5: Mesh Normalization

Before computing target losses, source and target vertices are normalized into a shared unit box:

```text
center = (vmax + vmin) / 2
scale  = 2 / max(vmax - vmin)
v_norm = (v - center) * scale
```

Why this matters:

- Loss magnitudes become comparable across animals.
- Chamfer thresholds and weights are more stable.
- Large animals and small animals can be optimized with the same recipe.

What to say:

> The losses operate in a normalized coordinate system. That makes the hyperparameters meaningful across horse-to-giraffe, panda-to-bear, elephant-to-giraffe, and other pairs.

### Slide 6: Global Chamfer Loss

The global Chamfer loss compares the whole deformed source to the whole target:

```text
L_global
  = mean_{x in S(V(theta))} min_{y in S(V_t)} ||x - y||_2^2
  + mean_{y in S(V_t)} min_{x in S(V(theta))} ||y - x||_2^2
```

Intuition:

- The first direction pulls the source onto the target.
- The second direction makes the source cover the target.

Strength:

- Good for coarse whole-object alignment.
- Useful when there is no semantic information.

Weakness:

- It is semantically blind.
- A leg can match a neck if it is spatially nearby.
- For cross-animal deformation, nearest-neighbor geometry alone can produce anatomically wrong correspondences.

Current status:

- Implemented in the pipeline.
- Set to `0.0` in the final hard PartField-only best runs.

What to say:

> Global Chamfer is a useful baseline, but for animals it is not enough. It does not know which geometry is a neck, leg, torso, head, or tail.

### Slide 7: Local Semantic Chamfer Loss

The main loss is local bucket Chamfer.

For each semantic bucket `k`, define:

```text
S_k(theta) = { v_i(theta) : l_s(i) = k }
T_k        = { u_j        : l_t(j) = k }
```

Then compute Chamfer only inside matching buckets:

```text
C_k
  = alpha mean_{x in S_k(theta)} min_{y in T_k} ||x - y||_2^2
  + beta  mean_{y in T_k}        min_{x in S_k(theta)} ||y - x||_2^2
```

Final PartField loss:

```text
L_partfield = (1 / |A|) sum_{k in A} C_k
```

where `A` is the set of active buckets. A bucket is active only if both source and target contain enough vertices in that bucket.

Current hyperparameters:

```text
alpha = 0.35   source-to-target bucket weight
beta  = 1.0    target-to-source bucket weight
points per bucket = 512
minimum points per bucket = 12
```

Why asymmetric Chamfer helps:

- The target-to-source direction has full weight, so the deformed mesh is pushed to cover the target anatomy.
- The source-to-target direction has lower weight, so source vertices are less aggressively pulled into bad nearest-neighbor matches.
- This reduces spikes when the source and target have different proportions.

What to say:

> This is the key idea. I am not comparing every source vertex to every target vertex. I only compare source head to target head, source torso to target torso, source legs to target legs, and so on.

### Slide 8: Why Hard Buckets Work

In the current best recipe:

```text
partfield_guidance_mode = hard
partfield_hard_weight   = 1.0
partfield_soft_weight   = 0.0
labels_aligned          = true
```

Hard bucket guidance means:

- Labels are assigned before optimization.
- Labels are frozen during optimization.
- A source vertex keeps its semantic identity while its position moves.
- Bucket `k` in the source only matches bucket `k` in the target.

This is especially strong for DenseCorr3D because those meshes have supervised annotated correspondences.

What to say:

> The hard labels turn the deformation into a semantically constrained geometry problem. The optimizer can still move the source shape freely, but it cannot cheat by matching the wrong anatomical region.

### Slide 9: DenseCorr3D as the Quantitative Core

DenseCorr3D meshes are the most important for numbers because they come from a supervised annotated animal dataset.

Important DenseCorr3D animals:

- Elephant: `external/DenseCorr3D/animals/2d6b3_toy_animals_009`
- Moose: `external/DenseCorr3D/animals/1d6d1_toy_animals_015`
- Giraffe: `external/DenseCorr3D/animals/34fb4_toy_animals_019`
- Panther: `external/DenseCorr3D/animals/071b8_toy_animals_017`
- Bear: `external/DenseCorr3D/animals/96615_toy_animals_018`
- Cheetah: `external/DenseCorr3D/animals/bdfd0_toy_animals_016`

Why DenseCorr3D matters:

- Labels are supervised, not guessed from generated images.
- Bucket identities can be aligned across animals.
- Quantitative part-level evaluation becomes meaningful.
- Failures can be diagnosed using bucket counts and part support.

What to say:

> DenseCorr3D is my strongest evaluation setting because the semantic structure is trustworthy. That lets me claim more than just nice visual examples.

### Slide 10: Bucket Balance And The Giraffe Neck Failure

The local loss depends on enough vertices in each semantic bucket.

If one target part has too few vertices, then the loss gives weak geometric support for that part.

Failure example:

- Elephant to giraffe is difficult because the giraffe neck is long and thin.
- If the neck bucket has too few target vertices, the source deformation can make the neck too wide.
- The optimizer is receiving the right semantic signal, but the sampling density is not strong enough to constrain the shape locally.

Scientific explanation:

```text
If |T_neck| is small, then
  mean_{x in S_neck} min_{y in T_neck} ||x - y||^2
and
  mean_{y in T_neck} min_{x in S_neck} ||y - x||^2
provide sparse constraints.
```

Consequence:

- The part is semantically correct but geometrically underconstrained.
- The failure is interpretable as a data/support issue, not a random hallucination.

What to say:

> The failure modes are useful because they tell us exactly what the pipeline needs. For giraffe necks, the issue is not that the method has no idea what a neck is. It is that the neck bucket needs enough local geometric support.

### Slide 11: Jacobian Deformation Parameterization

The deformation is parameterized through local Jacobians rather than arbitrary independent vertex offsets.

For each face `f`, the deformation has a local Jacobian:

```text
J_f in R^{3 x 3}
```

The identity deformation is:

```text
J_f = I
```

The mesh vertices are reconstructed from the Jacobian field using the pipeline's deformation machinery.

Why this is important:

- It treats deformation as a local differential field.
- It encourages coherent surface transformations.
- It gives direct access to smoothness and outlier regularizers.

What to say:

> Instead of only saying where each vertex should go, the method controls how local surface patches transform. That makes the deformation more structured.

### Slide 12: Jacobian Neighbor Smoothness

The active Jacobian regularizer is neighbor smoothness:

```text
L_jacobian_neighbor
  = mean_{(f,g) in E_F} ||J_f - J_g||_F^2
```

where `E_F` is the set of neighboring face pairs.

Current weight:

```text
lambda_jn = 1000
```

Purpose:

- Nearby faces should deform similarly.
- Prevents isolated face-level distortions.
- Makes the deformation field smooth over the mesh surface.

What to say:

> This is one of the main anti-artifact terms. If one face wants to rotate or scale very differently from its neighbors, the loss penalizes it.

### Slide 13: Identity Jacobian Regularizer

The code also supports an identity Jacobian prior:

```text
L_identity_jacobian
  = mean_f ||J_f - I||_F^2
```

Current final weight:

```text
lambda_jac = 0
```

Why it is disabled in the strongest final runs:

- Cross-animal deformations require large shape changes.
- A strong identity prior can over-constrain the deformation.
- Neighbor smoothness is more useful because it allows large but coherent transformations.

What to say:

> I keep the identity prior available, but the best current runs do not use it. For cross-animal deformation, we want the mesh to change a lot, just not chaotically.

### Slide 14: Jacobian Outlier Regularizer

The code also supports a higher-order outlier penalty:

```text
L_jacobian_outlier
  = mean_f ||J_f - I||_F^p
```

with:

```text
p = 4
```

Current final weight:

```text
lambda_out = 0
```

Purpose if enabled:

- Punishes extreme local transformations more strongly than a quadratic prior.
- Useful when a few faces become unstable.

What to say:

> This is available as an emergency brake for extreme local deformations, but the final recipe did not need it once the semantic Chamfer and neighbor smoothing were stable.

### Slide 15: Edge Displacement Jump Regularizer

The active edge regularizer penalizes discontinuous motion across mesh edges.

Let:

```text
d_i(theta) = v_i(theta) - v_i^0
```

be the displacement of vertex `i` from its original source position.

For each mesh edge `(i,j)`:

```text
r_ij
  = ||d_i(theta) - d_j(theta)||_2
    / ||v_i^0 - v_j^0||_2
```

Only jumps above a threshold are penalized:

```text
L_jump
  = mean_{(i,j) in E} [max(0, r_ij - tau)]^2
  + eta max_{(i,j) in E} [max(0, r_ij - tau)]^2
```

Current hyperparameters:

```text
lambda_jump = 500
tau         = 1.2
eta         = 2.0
```

Purpose:

- Prevents neighboring vertices from moving in sharply different directions.
- Reduces spikes and local tearing.
- Complements Jacobian neighbor smoothing at the vertex-displacement level.

What to say:

> The edge jump term is the spike guard. It allows large deformations, but it does not allow adjacent vertices to suddenly separate in displacement space.

### Slide 16: Edge Stretch Regularizer

The code also supports edge stretch regularization:

```text
r_ij
  = ||v_i(theta) - v_j(theta)||_2
    / ||v_i^0 - v_j^0||_2
```

```text
L_edge_stretch
  = mean_{(i,j) in E} [max(0, r_ij - tau_edge)]^2
  + eta_edge max_{(i,j) in E} [max(0, r_ij - tau_edge)]^2
```

Current final weight:

```text
lambda_edge = 0
```

Available hyperparameters in the final configs:

```text
tau_edge = 1.5
eta_edge = 1.0
```

Why it is not active:

- Cross-animal deformation often needs significant stretching.
- The displacement jump term catches sharp local artifacts without forbidding smooth large-scale stretching.

What to say:

> Edge stretch is available, but for these transformations it can be too restrictive. The active recipe focuses on preventing discontinuities rather than preventing all stretching.

### Slide 17: Final Best Recipe

Current strongest recipe used in the final meshes:

```text
epochs                         = 4000
deformation_parameterization   = jacobian
target_mesh_partfield_weight   = 8000
target_mesh_chamfer_weight     = 0
partfield_guidance_mode        = hard
partfield_hard_weight          = 1
partfield_soft_weight          = 0
partfield_source_to_target     = 0.35
partfield_target_to_source     = 1.0
points_per_bucket              = 512
minimum_points_per_bucket      = 12
jacobian_neighbor_smooth       = 1000
edge_displacement_jump         = 500
edge_displacement_threshold    = 1.2
edge_displacement_max_weight   = 2.0
SDS                            = off
DINO image loss                = off
```

What to say:

> This is the recipe I would present as the current state of the pipeline. It is simple enough to explain scientifically, but strong enough to produce the best current deformations.

### Slide 18: Why Removing SDS Was A Good Choice

Previous MeshUp-style direction:

```text
mesh -> render images -> compare through diffusion / DINO guidance -> update mesh
```

Problem:

- The supervision was only as good as the generated images.
- Stable Diffusion XL was needed for acceptable images.
- That made the loop computationally expensive.
- Bad images produced bad geometry.
- 2D priors did not provide exact 3D correspondences.

Current direction:

```text
mesh -> direct 3D target losses -> part-aware geometric optimization
```

Why this is better:

- Directly optimizes the 3D target objective.
- Avoids stochastic 2D image supervision.
- Removes diffusion compute from the inner loop.
- Produces interpretable losses and failure modes.

What to say:

> The question changed from "can a diffusion model generate a good picture of the target?" to "can the source geometry match the target geometry under semantic constraints?" That is a much cleaner research problem.

### Slide 19: Why Removing DINO From The Inner Loop Was Reasonable

DINO helped as a semantic feature idea, but image-space DINO guidance had limitations:

- It depends on camera views and rendered appearances.
- It gives feature similarity in 2D, not exact 3D geometric correspondence.
- It can be noisy for thin structures, occlusions, and cross-animal differences.
- It adds computation and more hyperparameters.

The current pipeline keeps the useful idea, semantic structure, but moves it into 3D:

```text
DINO/image-space semantics
  -> replaced by PartField/DenseCorr3D semantic buckets
```

What to say:

> I did not abandon semantics. I moved semantics from an unstable image-space loss into explicit 3D buckets that can be inspected, counted, and evaluated.

### Slide 20: Evaluation Metrics

The evaluation pipeline can measure both target fit and deformation quality.

Global target fit:

```text
Chamfer L2
Chamfer L2 squared
Hausdorff L2
source-to-target distance
target-to-source distance
normal consistency
precision / recall / F-score at unit-box thresholds
```

Part-level target fit:

```text
per-part Chamfer
per-part F-score
per-part normal consistency
mean part Chamfer
active part count
```

Source distortion:

```text
edge length relative mean / p95 / max
Laplacian delta mean / p95
```

What to say:

> I can evaluate whether the result looks like the target, whether each semantic part matches, and whether the source mesh was distorted too violently.

### Slide 21: Strong Qualitative Examples

Use the final meshes as visual evidence:

- Horse to giraffe:
  `jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/colored_meshes/mesh_final_position.ply`

- Panda to bear:
  `jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/panda3_to_bear_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_panda3_to_bear_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5817242/colored_meshes/mesh_final_position.ply`

- Fox to cat:
  `jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/fox_to_cat2_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_fox_to_cat2_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/colored_meshes/mesh_final_position.ply`

- Panda to panda:
  `jobs_with_target_guidance/cross_animal_spike_runs/outputs/sam3d_selected_best_asym035_jump500_4000/panda2_to_panda3_sam3d_selected_pf09_4000_hard_partfield_chamfer_only_panda2_to_panda3_pf09_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/colored_meshes/mesh_final_position.ply`

What to say:

> These are the visual examples showing that the method works even across different animal families. The panda-to-panda result is useful as an easier same-category sanity check.

### Slide 22: Chain Deformation Example

Panther to horse to giraffe chain:

- Panther to horse:
  `/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2/jobs_with_target_guidance/cross_animal_spike_runs/outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/panther_to_horse2_chain_face_x_pf06_4000_hard_partfield_chamfer_only_panther_to_horse2_pf06_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829892/mesh_final/mesh.obj`

- Horse to dense giraffe:
  `/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2/jobs_with_target_guidance/cross_animal_spike_runs/outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/horse2_to_dense_giraffe_chain_face_x_pf08_4000_hard_partfield_chamfer_only_horse2_to_dense_giraffe_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829931/mesh_final/mesh.obj`

Why this is interesting:

- Shows compositional deformation.
- Tests whether an intermediate animal can make a hard transformation easier.
- Useful for explaining progressive shape morphing.

What to say:

> The chain experiment is interesting because it asks whether very hard transformations can be decomposed into more anatomically plausible steps.

### Slide 23: Main Failure Modes

Failure mode 1: bucket imbalance

- Some target parts have too few vertices.
- Thin structures like giraffe necks become underconstrained.
- The optimizer gets correct semantic direction but weak local shape detail.

Failure mode 2: topology mismatch

- The source topology stays fixed.
- If the target has a structure the source does not support well, the deformation can stretch or thicken the source geometry.

Failure mode 3: hard label errors

- Hard buckets are powerful, but they inherit mistakes from the labels.
- Wrong labels create wrong local attractions.

Failure mode 4: over-regularization versus under-regularization

- Too much regularization blocks real cross-animal shape change.
- Too little regularization allows spikes or local tearing.

What to say:

> The failure modes are not mysterious. They come from identifiable parts of the objective: label quality, bucket density, topology, and regularization balance.

### Slide 24: Ablation Story

Suggested ablation table:

```text
Experiment                         Expected result
---------------------------------  -----------------------------------------
Global Chamfer only                coarse shape, wrong part correspondences
PartField Chamfer only             better anatomy, possible spikes
PartField + j-neighbor             smoother local deformation
PartField + edge jump              fewer spikes and local discontinuities
PartField + both regularizers      best current recipe
SDS/DINO image guidance            higher compute, image-dependent failures
```

What to say:

> The ablation story is simple: global matching gives shape, semantic buckets give correct anatomy, and the regularizers make the deformation usable.

### Slide 25: What Makes This Scientifically Strong

The pipeline is strong because it is:

- Interpretable: every loss term has a geometric meaning.
- Ablatable: each term can be turned on or off.
- Quantitative: target fit and part fit can be measured.
- Supervised where it matters: DenseCorr3D provides annotated labels.
- Efficient compared with diffusion guidance: no SDXL in the optimization loop.
- Diagnostic: failures point to bucket support, topology, or regularization.

What to say:

> The contribution is not just that the meshes look better. The contribution is that the optimization is now explainable, measurable, and grounded in 3D geometry.

## Technical Deep Dive: Loss Terms

### 1. PartField / DenseCorr3D Bucket Chamfer

Definitions:

```text
V(theta): current source vertices
U: target vertices
l_s(i): source label for vertex i
l_t(j): target label for vertex j
K: number of semantic buckets
```

For each bucket:

```text
S_k = { V_i(theta) : l_s(i) = k }
T_k = { U_j        : l_t(j) = k }
```

A bucket is active if:

```text
|S_k| >= 12 and |T_k| >= 12
```

Per-bucket asymmetric Chamfer:

```text
C_k
  = 0.35 mean_{x in S_k} min_{y in T_k} ||x - y||_2^2
  + 1.00 mean_{y in T_k} min_{x in S_k} ||y - x||_2^2
```

Full hard bucket loss:

```text
L_partfield = mean_{k active} C_k
```

Final weighted contribution:

```text
8000 L_partfield
```

Implementation notes:

- Up to `512` vertices are sampled per bucket.
- Labels are frozen at initialization.
- Source and target labels are aligned in the final runs.
- In DenseCorr3D runs, labels come from supervised annotated animal data.

### 2. Global Chamfer

General form:

```text
L_global
  = mean_x min_y ||x - y||_2^2
  + mean_y min_x ||y - x||_2^2
```

Role:

- Coarse full-shape alignment.
- Useful baseline.

Final recipe:

```text
lambda_global = 0
```

Reason:

- Hard semantic bucket Chamfer gives more reliable correspondences for cross-animal deformation.

### 3. Jacobian Neighbor Smoothness

Each face has a local deformation Jacobian:

```text
J_f in R^{3 x 3}
```

Adjacent face pairs:

```text
(f,g) in E_F
```

Loss:

```text
L_jneighbor = mean_{(f,g) in E_F} ||J_f - J_g||_F^2
```

Final weighted contribution:

```text
1000 L_jneighbor
```

Purpose:

- Keeps neighboring surface patches deforming coherently.
- Prevents high-frequency deformation noise.
- Allows large deformations if they are spatially smooth.

### 4. Identity Jacobian Prior

Loss:

```text
L_identity = mean_f ||J_f - I||_F^2
```

Final recipe:

```text
lambda_identity = 0
```

Reason:

- Strong identity preservation can prevent large cross-animal transformations.
- Neighbor smoothness was more useful in the final recipe.

### 5. Jacobian Outlier Prior

Loss:

```text
L_outlier = mean_f ||J_f - I||_F^4
```

Final recipe:

```text
lambda_outlier = 0
```

Role:

- Penalizes extreme face transformations.
- Available for unstable cases.

### 6. Edge Displacement Jump

Original source vertex:

```text
v_i^0
```

Current vertex:

```text
v_i(theta)
```

Displacement:

```text
d_i = v_i(theta) - v_i^0
```

Normalized displacement jump across edge `(i,j)`:

```text
r_ij = ||d_i - d_j||_2 / ||v_i^0 - v_j^0||_2
```

Loss:

```text
L_jump
  = mean_{(i,j) in E} [max(0, r_ij - 1.2)]^2
  + 2.0 max_{(i,j) in E} [max(0, r_ij - 1.2)]^2
```

Final weighted contribution:

```text
500 L_jump
```

Purpose:

- Prevents spikes.
- Penalizes neighboring vertices that suddenly move differently.
- Keeps displacement field locally continuous.

### 7. Edge Stretch

Loss:

```text
r_ij = ||v_i(theta) - v_j(theta)||_2 / ||v_i^0 - v_j^0||_2
```

```text
L_edge_stretch
  = mean_{(i,j) in E} [max(0, r_ij - 1.5)]^2
  + 1.0 max_{(i,j) in E} [max(0, r_ij - 1.5)]^2
```

Final recipe:

```text
lambda_edge_stretch = 0
```

Reason:

- Large cross-animal deformation requires stretch.
- Displacement jump catches the more damaging artifact: local discontinuity.

## What To Emphasize To The Committee

### Main Scientific Contribution

The method combines semantic correspondence and differential deformation regularization:

```text
semantic local target matching + smooth Jacobian field + spike suppression
```

This gives an interpretable alternative to image-prior deformation.

### Why The Idea Is Strong

The pipeline is not just producing pretty examples. It is an optimization framework where every term is meaningful:

- Chamfer tells the mesh where the target geometry is.
- Buckets tell the mesh which target geometry is semantically compatible.
- Jacobian neighbors tell nearby faces to deform similarly.
- Edge jumps prevent local vertex displacement discontinuities.
- DenseCorr3D labels provide supervised semantic structure.

### Best One-Minute Explanation

> I deform a source mesh into a target mesh using a geometry-first optimization. The key loss is a semantic Chamfer loss: instead of matching the whole source to the whole target blindly, I split both meshes into aligned semantic buckets and compute Chamfer only within corresponding buckets. Then I regularize the deformation field using neighboring face Jacobians and an edge displacement jump penalty, which prevents spikes. In the final recipe, the active objective is PartField/DenseCorr3D bucket Chamfer plus Jacobian-neighbor smoothing plus edge-jump regularization. This removes the need for Stable Diffusion or image-space DINO in the optimization loop and gives a more interpretable, measurable 3D pipeline.

## Suggested Figure Set

Figure 1: Pipeline diagram

```text
source mesh -> semantic labels -> bucket Chamfer -> Jacobian optimization -> final mesh
target mesh -> semantic labels -----------^
```

Figure 2: Global Chamfer vs local bucket Chamfer

Show:

- Global Chamfer nearest neighbor links crossing between wrong animal parts.
- Bucket Chamfer links only inside same semantic label.

Figure 3: Deformation regularization

Show:

- Neighboring faces with similar Jacobians.
- Edge jump penalty preventing adjacent vertices from separating too sharply.

Figure 4: DenseCorr3D examples

Show:

- Panther to cheetah
- Cheetah to panther
- Elephant to giraffe
- Giraffe to elephant
- Elephant to moose

Figure 5: Failure case

Show:

- Elephant to giraffe neck distortion.
- Add bucket-count explanation: too few vertices in the neck bucket underconstrain the thin structure.

## Suggested Results Table

Use one table for qualitative runs:

```text
Pair              Data source    Buckets    Result summary
---------------- -------------  ---------  --------------------------------
horse -> giraffe  SAM3D/PartField  8        strong cross-family shape change
panda -> bear     SAM3D/PartField  7        strong same-body-family transfer
fox -> cat        SAM3D/PartField  7        compact quadruped transfer
panda -> panda    SAM3D/PartField  9        sanity check, same category
panther -> horse  chain            6        intermediate chain step
horse -> giraffe  chain            8        final chain step
```

Use another table for DenseCorr3D quantitative runs:

```text
Pair                  Why important
--------------------  -------------------------------------------------
panther -> cheetah    supervised labels, similar topology, good metric test
cheetah -> panther    reverse-direction consistency
elephant -> giraffe   hard anatomy change, neck/trunk mismatch stress test
giraffe -> elephant   reverse hard anatomy change
elephant -> moose     large quadruped with different head/antler structure
bear -> panther       body-shape and limb proportion transfer
bear -> elephant      large morphology gap
```

## Honest Limitations To Present

1. Hard labels are only as good as the label source.
2. Bucket imbalance can underconstrain thin parts.
3. Fixed source topology limits extreme anatomical changes.
4. Some shape changes need remeshing or better target sampling.
5. Same loss weights may not be optimal for every animal pair.

Strong framing:

> These limitations are useful because they point to concrete next steps: bucket balancing, better sampling, label refinement, and topology-aware deformation.

## Next Steps

Short-term:

- Run full quantitative evaluation on DenseCorr3D pairs.
- Report global Chamfer, part Chamfer, F-score, normal consistency, and distortion metrics.
- Add bucket count tables beside each result.
- Compare against global-Chamfer-only and no-regularizer baselines.

Medium-term:

- Improve bucket balancing for thin parts like giraffe necks.
- Add adaptive per-bucket sampling or per-bucket weights.
- Explore soft/hybrid bucket matching for ambiguous regions.
- Add topology-aware remeshing when source and target anatomy differ too much.

## Code Evidence Used For This Outline

Main implementation files:

- `jobs_with_target_guidance/partfield_chamfer.py`
- `loop_tracked.py`
- `jobs_with_target_guidance/evaluate_target_pipeline.py`
- `jobs_with_target_guidance/part_aware_chamfer.py`

Representative final configs:

- `horse2_to_giraffe4.../config.yml`
- `panda3_to_bear.../config.yml`
- `panther_to_horse2.../config.yml`

DenseCorr3D run evidence:

- `jobs_with_target_guidance/cross_animal_spike_runs/scripts/run_densecorr3d_requested_animal_pair.sh`
- `jobs_with_target_guidance/cross_animal_spike_runs/reports/densecorr3d_bucketbalanced_requested_2026-07-02.md`
- `jobs_with_target_guidance/cross_animal_spike_runs/reports/densecorr3d_srcstable_rerun_2026-07-02.md`

