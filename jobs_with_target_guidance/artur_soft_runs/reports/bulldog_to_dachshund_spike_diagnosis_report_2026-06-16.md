# Bulldog to Dachshund Mesh Deformation: Spike Diagnosis and Fix Attempts

Date: 2026-06-16

## 1. Short summary

We are deforming a bulldog-like source mesh into a dachshund-like target mesh using the MeshUp codebase, but without SDS/diffusion. The current pipeline uses:

- PartField to separate source and target meshes into semantic regions.
- Chamfer loss in 3D to pull the source mesh toward the target mesh.
- Jacobian-based deformation, so the trainable variables are local deformation Jacobians, not independent free vertices.
- Adam as the optimizer.

The best original result looked mostly like a dachshund, but two visible spike problems appeared:

- A spike/thin stretched region around the right/back leg.
- A spike/thin stretched region around the left ear/top of head.

The diagnosis showed that the spike vertices were not random and were not simply unmatched. They were being pulled by the source-to-target part of Chamfer toward specific target vertices inside the matching PartField bucket. However, many of these same source spike vertices were not selected by the reverse target-to-source Chamfer term. In simple words:

> The spike vertices are chasing target points, but the target mesh does not really need those exact source vertices to cover it.

This is a typical many-to-one nearest-neighbor Chamfer problem. Chamfer does not enforce a clean one-to-one correspondence, so several source vertices can chase the same target point or boundary area.

The best practical fix so far is the local displacement-jump regularizer, especially with the asymmetric PartField Chamfer test:

- `jump500` improved the spikes while keeping the dachshund shape.
- `asym_jump500` made a small additional improvement in some spike metrics by weakening source-to-target PartField pull.

Recommended run to show:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590
```

Final mesh:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/mesh_final/mesh.obj
```

Correspondence visualization:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/displacement_viz/chamfer_correspondence_probe/source_target_partfield_match_lines.ply
```

## 2. What the pipeline is doing

The goal is to deform a source mesh into the shape of a target mesh.

In this experiment:

- Source mesh: bulldog-like dog.
- Target mesh: dachshund-like dog.
- We removed SDS/diffusion from the original MeshUp pipeline.
- We are doing direct 3D deformation.
- PartField gives semantic regions/parts on the source and target.
- Chamfer loss compares points in 3D between the current source mesh and the target mesh.
- The source mesh is optimized for 2500 epochs in the dev runs.

The important point is that the target mesh does not move. The source mesh moves. At every optimization step, the current deformed source mesh is compared to the fixed target mesh.

## 3. What Chamfer loss is, in simple words

Chamfer loss compares two point clouds or meshes by nearest neighbors.

Let:

- `S` be points sampled from the current source mesh.
- `T` be points sampled from the target mesh.

Chamfer has two directions.

### 3.1 Source-to-target term

For each source point, find the closest target point.

Simple example:

```text
source point s_1 -> closest target point t_7
source point s_2 -> closest target point t_7
source point s_3 -> closest target point t_20
```

Then compute distances and average them.

This term asks:

> Is every source point close to some target point?

This is useful because it prevents source vertices from staying far away. But it can also cause problems: if a source vertex has no good semantic one-to-one match, it may chase a target boundary point or corner.

### 3.2 Target-to-source term

For each target point, find the closest source point.

Simple example:

```text
target point t_1 -> closest source point s_8
target point t_2 -> closest source point s_8
target point t_3 -> closest source point s_15
```

This term asks:

> Is every target point covered by some source point?

This is useful because it prevents missing target regions. For example, if the dachshund body is long, the target-to-source term asks the source mesh to cover that long body.

### 3.3 Symmetric Chamfer

The usual symmetric Chamfer is:

```text
Chamfer(S, T) =
    average distance from each source point to nearest target point
  + average distance from each target point to nearest source point
```

In the original PartField hard Chamfer, both directions had equal weight:

```text
source-to-target weight = 1.0
target-to-source weight = 1.0
```

In the newest asymmetric test, we used:

```text
source-to-target weight = 0.35
target-to-source weight = 1.0
```

The idea was:

> Let the target shape still request coverage strongly, but reduce the tendency for extra source vertices to individually chase target boundary points.

## 4. What PartField changes

Plain Chamfer compares all source points to all target points. That can match a source leg to a target ear if they are geometrically close enough.

PartField reduces this by dividing the meshes into semantic buckets.

In the hard PartField Chamfer used here:

- Source bucket 1 compares only to target bucket 1.
- Source bucket 8 compares only to target bucket 8.
- And so on.

This is better than global Chamfer alone because it limits wrong cross-part matching.

However, PartField bucket matching is still not a true one-to-one correspondence. Inside one bucket, Chamfer still uses nearest neighbors. So many source vertices can still chase one target vertex inside the same part.

That is exactly what we observed in the spike diagnostics.

## 5. Where Adam gets involved

Adam is the optimizer. It does not decide correspondences itself. The correspondences come from nearest-neighbor matching inside the Chamfer loss.

The loop is:

1. Start with current source mesh deformation.
2. Compute current source vertex positions from the trainable Jacobians.
3. Compute Chamfer loss against the target mesh.
4. Compute regularization losses, such as Jacobian neighbor smoothness and spike guards.
5. Add all losses into one total loss.
6. PyTorch computes gradients of the total loss.
7. Adam updates the trainable deformation variables.
8. The mesh changes.
9. At the next epoch, nearest-neighbor Chamfer matches are recomputed from the new mesh positions.

In these runs, the deformation representation was:

```text
DEFORMATION_PARAMETERIZATION=jacobian
```

So Adam updates per-face Jacobians, not directly free vertex coordinates. The vertices move because the mesh is reconstructed from those Jacobians.

Adam can contribute to overshoot in the practical sense that it follows gradients from the loss. If a source vertex receives a strong gradient toward a target point, Adam can keep pushing that local region. But the root cause is not Adam alone. The root cause is that the Chamfer objective gives a strong nearest-neighbor pull without a one-to-one correspondence constraint.

## 6. Why spikes can happen

The spike behavior can happen because:

1. Chamfer is many-to-one.
   Several source vertices can all choose the same target vertex.

2. Source-to-target matching can pull unnecessary source vertices.
   Even if the target mesh is already covered by other source vertices, the source-to-target term still asks every source vertex to find some target point.

3. The reverse direction may not use the spike vertex.
   In the diagnostic table this appears as:

```text
PF tgt->src count = 0
```

That means no sampled target point chose that source spike vertex as its nearest source point in the reverse PartField Chamfer direction.

4. The nearest neighbor can be a boundary/corner point in the target bucket.
   If a source vertex is pulled toward a boundary point and its neighbors are not pulled the same way, it creates a thin local stretch.

5. Strong PartField weight can amplify this.
   We used:

```text
PARTFIELD_CHAMFER_WEIGHT=8000
GLOBAL_CHAMFER_WEIGHT=750
```

So the PartField Chamfer has a strong influence.

## 7. Diagnostic method

I added and used:

```text
jobs_with_target_guidance/analyze_chamfer_spike_correspondences.py
```

This script does three things:

1. Finds or analyzes suspicious spike vertices.
2. Recomputes which target vertex each suspicious source vertex is closest to inside its PartField bucket.
3. Exports tables and PLY visualizations.

For each suspicious source vertex, it reports:

- Source vertex ID.
- PartField bucket ID.
- Spike score.
- Nearest target vertex under PartField Chamfer.
- Distance to that target vertex.
- Whether the reverse target-to-source term selected this source vertex.
- Global nearest target vertex.

The important output files are:

```text
chamfer_spike_correspondences.md
chamfer_spike_correspondences.csv
chamfer_spike_correspondences.json
source_spikes_marked.ply
target_matches_marked.ply
source_target_partfield_match_lines.ply
```

## 8. What source_target_partfield_match_lines.ply shows

This file is the easiest visualization to explain.

It contains:

- The deformed source mesh, shifted to one side.
- The target mesh, shifted to the other side.
- PartField colors on both meshes.
- White highlighted source spike vertices.
- White highlighted target match vertices.
- Yellow line segments connecting each suspicious source vertex to the target vertex it is currently matched to by the PartField source-to-target nearest-neighbor rule.

Important:

This does not show a manually chosen correspondence. It shows what the Chamfer loss sees at the final mesh, using the same PartField bucket restriction and vertex sampling logic as the training loss.

How to explain it:

> Each yellow line says: this source spike vertex is being pulled toward this target point by the PartField Chamfer source-to-target term.

If several yellow lines go to the same target point, that means many source vertices are chasing one target point. That is evidence of many-to-one Chamfer behavior.

If a source spike vertex has `PF tgt->src count = 0`, that means the reverse direction does not really need that source vertex. In simple words:

> The source vertex wants the target point, but the target point does not depend on that source vertex.

That is a strong sign that the spike is caused by source-to-target Chamfer pull rather than a true necessary target coverage requirement.

## 9. Original diagnosis numbers

Original best run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jneighbor_1000/bulldog_to_dachshund_artur_pf_chamfer_jneighbor1000_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_2500ep_4889324
```

Original best mesh:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jneighbor_1000/bulldog_to_dachshund_artur_pf_chamfer_jneighbor1000_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_2500ep_4889324/mesh_final/mesh.obj
```

The original result looked like a dachshund overall, but had visible spikes.

Key spike correspondences:

| source vertex | area | bucket | original score | original PF target | original PF distance | reverse PF count |
|---:|---|---:|---:|---:|---:|---:|
| 198 | right/back leg spike | 1 | 5.8806 | 968 | 0.2603 | 0 |
| 38 | right/back leg spike | 1 | 2.7195 | 968 | 0.1807 | 0 |
| 546 | right/back leg spike | 1 | 1.9706 | 968 | 0.1189 | 0 |
| 717 | right/back leg spike | 1 | 1.5142 | 968 | 0.1203 | 0 |
| 3368 | left ear spike | 8 | 2.6558 | 2445 | 0.1319 | 0 |
| 3572 | left ear neighbor | 8 | 0.8837 | 2600 | 0.0445 | 0 |
| 3365 | left ear neighbor | 8 | 0.7202 | 2600 | 0.0363 | 0 |
| 3701 | left ear neighbor | 8 | 0.6742 | 2826 | 0.0164 | 1 |

Interpretation:

- The leg spike vertices were mostly aiming at target vertex `968` or nearby target vertex `717`.
- The ear spike vertex `3368` was aiming at target vertex `2445`.
- Many reverse counts were `0`.
- This supports the idea that these are one-way source-to-target Chamfer pulls, not clean semantic correspondences.

## 10. Fix attempts and results

### 10.1 Strict containment plus edge-stretch guard

Run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_containment_edge_20/bulldog_to_dachshund_artur_pf_chamfer_contain20_edge_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_contain20_edge_2500ep_5032728
```

What we tried:

- PartField containment weight `20`.
- Edge-stretch guard.
- Strong Jacobian regularization and outlier regularization.

Important settings:

```text
PARTFIELD_CONTAINMENT_WEIGHT=20
EDGE_STRETCH_WEIGHT=1000
EDGE_STRETCH_THRESHOLD=1.25
JACOBIAN_REG_WEIGHT=1000
JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000
JACOBIAN_OUTLIER_WEIGHT=500
```

Why we tried it:

The visible issue looked like local vertices going too far away. So we tried to keep vertices inside the matching target PartField region and prevent edges from stretching too much.

Result:

- Some spike scores improved numerically.
- But the dog looked much less like a dachshund and more like a bulldog.
- The visible ear spike was still bad.

Conclusion:

This was too strict. It attacked deformation itself, not only the spike. A dachshund deformation needs coherent shape changes, including body elongation and part movement. The edge-stretch guard made the whole mesh too stiff.

This run is not recommended as the final fix.

### 10.2 Balanced Sinkhorn PartField transport

Run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_balanced/bulldog_to_dachshund_artur_pf_chamfer_balanced_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_balanced_sinkhorn_2500ep_5036551
```

What we tried:

- A balanced transport version of PartField bucket matching.
- The motivation was to reduce many-to-one matching by giving target points a more balanced capacity.

Result:

This did not work well.

Important scores:

| source vertex | original score | balanced score |
|---:|---:|---:|
| 198 | 5.8806 | 7.3668 |
| 3368 | 2.6558 | 4.6461 |

It also shrank/collapsed the mesh scale compared to the good dachshund result.

Conclusion:

Balanced transport was a good hypothesis, but this implementation was not a good practical fix for this case. It likely changed the global deformation too much and did not solve the local spike behavior.

### 10.3 Local displacement-jump penalty

Run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jump_500/bulldog_to_dachshund_artur_pf_chamfer_jump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_jump500_2500ep_5036806
```

Final mesh:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jump_500/bulldog_to_dachshund_artur_pf_chamfer_jump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_jump500_2500ep_5036806/mesh_final/mesh.obj
```

What we tried:

Instead of preventing every edge from getting long, we penalized only isolated differences in neighboring vertex motion.

In simple words:

> Neighboring vertices are allowed to move, but one vertex should not run away from its neighbors.

Important settings:

```text
EDGE_DISPLACEMENT_JUMP_WEIGHT=500
EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.2
EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.0
JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000
PARTFIELD_CONTAINMENT_WEIGHT=0
EDGE_STRETCH_WEIGHT=0
```

Why this is better than edge-stretch:

- Edge-stretch says: this edge cannot become much longer.
- Displacement-jump says: the endpoints of this edge should not move very differently.

So a whole region can still move into dachshund shape. But an isolated vertex is discouraged from becoming a spike.

Result:

| source vertex | original score | jump500 score |
|---:|---:|---:|
| 198 | 5.8806 | 2.2930 |
| 38 | 2.7195 | 1.0690 |
| 546 | 1.9706 | 0.7274 |
| 3368 | 2.6558 | 0.7599 |
| 3572 | 0.8837 | 0.1745 |
| 3365 | 0.7202 | 0.5451 |
| 3701 | 0.6742 | 0.3169 |

Conclusion:

This was the first successful fix. It reduced spike behavior while keeping the dachshund-like shape.

### 10.4 Asymmetric PartField Chamfer plus displacement-jump penalty

Run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590
```

Final mesh:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/mesh_final/mesh.obj
```

What we tried:

We kept the successful local displacement-jump penalty, and also changed hard PartField Chamfer direction weights:

```text
source-to-target PartField weight = 0.35
target-to-source PartField weight = 1.0
```

Why we tried it:

The diagnosis suggested that the spike vertices were often being pulled by the source-to-target direction, while the target-to-source direction did not need them. So we reduced the source-to-target pressure while preserving target coverage.

Result:

| source vertex | original score | jump500 score | asym_jump500 score |
|---:|---:|---:|---:|
| 198 | 5.8806 | 2.2930 | 2.1411 |
| 38 | 2.7195 | 1.0690 | 1.1988 |
| 546 | 1.9706 | 0.7274 | 0.6911 |
| 717 | 1.5142 | 0.3289 | 0.5084 |
| 3368 | 2.6558 | 0.7599 | 0.6460 |
| 3572 | 0.8837 | 0.1745 | 0.2415 |
| 3365 | 0.7202 | 0.5451 | 0.6648 |
| 3701 | 0.6742 | 0.3169 | 0.3257 |

The result is mixed but useful:

- Vertex `198`, the main leg spike, improved slightly compared to jump500.
- Vertex `3368`, the main ear spike, improved slightly compared to jump500.
- A few neighboring vertices became slightly worse.
- The visual dachshund shape was preserved.

Conclusion:

The asymmetric Chamfer test supports the diagnosis: reducing source-to-target pressure helps the main spike vertices. However, the improvement over jump500 is small. The most reliable fix so far is the displacement-jump regularizer, with asymmetric PartField Chamfer as a useful additional experiment.

## 11. Latest asym_jump500 correspondences

For the latest run, the analyzed spike vertices correspond to:

| source vertex | area | PartField target | PF distance | reverse PF count |
|---:|---|---:|---:|---:|
| 198 | right/back leg | 717 | 0.1731 | 0 |
| 38 | right/back leg | 717 | 0.1384 | 0 |
| 546 | right/back leg | 717 | 0.1116 | 0 |
| 717 | right/back leg | 968 | 0.1139 | 0 |
| 3368 | left ear/top | 2671 | 0.0527 | 0 |
| 3572 | left ear/top | 2671 | 0.0176 | 1 |
| 3365 | left ear/top | 2671 | 0.0191 | 0 |
| 3701 | left ear/top | 2994 | 0.0094 | 2 |

Interpretation:

- The leg cluster is now mostly aiming at target vertex `717`.
- The ear cluster is now aiming at target vertices around `2671`, `2600`, and `2994`.
- The distances are much smaller for the ear cluster than in the original run.
- Some reverse counts are still `0`, so the one-way Chamfer issue is reduced but not fully solved.

## 12. What this tells us scientifically

The experiments show:

1. The spikes are not random numerical noise.
   They correspond to specific target vertices under the PartField Chamfer nearest-neighbor rule.

2. The spikes are not simply because PartField failed completely.
   The target points are usually in the same PartField bucket. The issue is inside-bucket many-to-one Chamfer matching.

3. Hard global shape constraints can make the mesh worse.
   Containment and edge-stretch reduced some metrics but hurt the dachshund shape and did not reliably remove visible spikes.

4. A local motion regularizer is more appropriate.
   The displacement-jump penalty attacks isolated runaway vertices while allowing the whole dog to deform.

5. Reducing source-to-target Chamfer pressure helps the main spike vertices slightly.
   This supports the idea that source-to-target nearest-neighbor pull contributes to the spikes.

## 13. Recommended explanation to supervisor

I would explain it like this:

> The deformation is mostly good, but Chamfer loss can create spikes because it does not know true vertex-to-vertex correspondences. It only uses nearest neighbors. We diagnosed the spike vertices and found that they are being pulled toward specific target vertices inside the matching PartField bucket. Many of these spike vertices are not used by the reverse target-to-source Chamfer direction, so they are likely unnecessary source-to-target matches. We visualized this using `source_target_partfield_match_lines.ply`, where each yellow line shows which target point a spike vertex is chasing. Then we tested several fixes. Strict containment and edge-stretch made the dog too rigid and less dachshund-like. Balanced transport did not work. The best fix so far is a local displacement-jump penalty, which prevents individual vertices from moving very differently from their neighbors. We also tested asymmetric Chamfer by reducing the source-to-target PartField weight, which slightly improved the main spike vertices and supports the diagnosis.

## 14. Suggested next steps

The next improvements I would try are:

1. Use displacement-jump as the default spike guard.
   It is currently the best practical fix.

2. Tune asymmetric Chamfer weights.
   We tried `0.35 / 1.0`. Good next values:

```text
source-to-target = 0.20, target-to-source = 1.0
source-to-target = 0.50, target-to-source = 1.0
```

3. Add a soft one-to-many correspondence instead of hard nearest neighbor inside each PartField bucket.
   This would make a source vertex move toward a local target region, not a single target vertex.

4. Add a repulsion or capacity term only for problematic buckets.
   This would prevent many source vertices from collapsing to the same target point.

5. Use a time schedule.
   Early training can use stronger target-to-source coverage to get the global dachshund shape. Later training can increase local regularization to suppress spikes.

## 15. Important files

Original best result:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jneighbor_1000/bulldog_to_dachshund_artur_pf_chamfer_jneighbor1000_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_2500ep_4889324/mesh_final/mesh.obj
```

Failed containment/edge run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_containment_edge_20/bulldog_to_dachshund_artur_pf_chamfer_contain20_edge_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_contain20_edge_2500ep_5032728/mesh_final/mesh.obj
```

Jump500 improved run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jump_500/bulldog_to_dachshund_artur_pf_chamfer_jump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_jump500_2500ep_5036806/mesh_final/mesh.obj
```

Asymmetric jump500 run:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/mesh_final/mesh.obj
```

Latest correspondence visualization:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/displacement_viz/chamfer_correspondence_probe/source_target_partfield_match_lines.ply
```

Latest correspondence table:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/displacement_viz/chamfer_correspondence_probe/chamfer_spike_correspondences.md
```

Latest final render grid:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/epoch_renders/epoch_02500/grid_all_views.png
```

Diagnostic script:

```text
jobs_with_target_guidance/analyze_chamfer_spike_correspondences.py
```

## 16. Code changes made

New local spike regularizer:

```text
edge_displacement_jump_weight
edge_displacement_jump_threshold
edge_displacement_jump_max_weight
```

Purpose:

> Penalize isolated neighboring-vertex displacement differences, while still allowing coherent regional deformation.

New asymmetric PartField Chamfer weights:

```text
partfield_source_to_target_weight
partfield_target_to_source_weight
```

Purpose:

> Reduce the source-to-target nearest-neighbor pull that can create unnecessary spikes, while preserving target-to-source coverage.

Default behavior is unchanged unless these values are explicitly set:

```text
edge_displacement_jump_weight = 0.0
partfield_source_to_target_weight = 1.0
partfield_target_to_source_weight = 1.0
```

