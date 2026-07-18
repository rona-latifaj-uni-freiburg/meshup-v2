# Semantic Vertex-Correspondence Bridge

Date: 2026-06-21

## Diagnosis

`topomatch_vcorr` works because the current cross-animal meshes share vertex
counts and face tensors, so the target can supervise each source vertex directly.
The weaker PartField-bucket Chamfer variants still solve a nearest-neighbor
problem inside large semantic buckets, which permits many-to-one pulls and local
spikes.

The bridge is to keep the direct vertex target loss, but replace identity-only
target indices with a dense semantic source-vertex -> target-vertex map.

## Research Signal

- PartField learns continuous 3D part features with cross-shape consistency,
  useful for co-segmentation and correspondence: https://arxiv.org/abs/2504.11451
- Zero-shot 3D correspondence uses semantic regions first, then refines to dense
  maps; hard regions alone are not the endpoint: https://arxiv.org/abs/2306.03253
- Smooth Shells and ZoomOut show that dense correspondence benefits from
  descriptor matches plus smooth/multiscale refinement:
  https://arxiv.org/abs/1905.12512 and https://arxiv.org/abs/1904.07865
- DINOv2 provides strong general visual features, but here the 3D PartField
  features are the safer primary descriptor because they avoid view aggregation
  ambiguities: https://arxiv.org/abs/2304.07193
- SAMPart3D is a plausible alternate part source, but it is still part-level;
  this change needs a dense vertex map: https://arxiv.org/abs/2411.07184

## Implemented Change

New module:

```text
jobs_with_target_guidance/semantic_vertex_correspondence.py
```

It builds a dense correspondence by:

- converting vertex-level or face-level PartField descriptors to vertices,
- matching in descriptor space with bbox-position and normal tie-breakers,
- applying optional soft/hard aligned PartField label constraints,
- computing confidence from similarity, top-k ambiguity, and mutual matching,
- optionally keeping topology identity when source/target faces already match
  unless semantic evidence is stronger,
- saving an NPZ cache and optional retargeted OBJ.

The training loop now supports:

```text
target_mesh_semantic_vertex_correspondence_weight
target_mesh_semantic_vertex_correspondence_warmup_epochs
semantic_vertex_correspondence_cache
semantic_vertex_correspondence_* matching controls
```

This loss is a weighted direct MSE from each current source vertex to its
semantic target vertex, so it uses the same high-signal mechanism as
`target_mesh_vertex_correspondence_weight` without requiring equal topology.

## Default Experiment

New launchers:

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_semantic_vcorr_array.sh
```

Outputs:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/semantic_vcorr_partfield
jobs_with_target_guidance/cross_animal_spike_runs/correspondences
```

Default recipe:

- `TARGET_VERTEX_CORRESPONDENCE_WEIGHT=0`
- `TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=20000`
- `SEMANTIC_VERTEX_CORRESPONDENCE_LABEL_FILTER=soft`
- `SEMANTIC_VERTEX_CORRESPONDENCE_TOPOLOGY_PRIOR_WEIGHT=0.20`
- global Chamfer and PartField Chamfer disabled
- vertex deformation parameterization

## Local Check

For `dachshund_to_cat`, the CPU correspondence/loss check produced:

```text
raw semantic vertex loss: 0.1500207335
unique target vertices: 3111 / 4712
mean correspondence weight: 0.313567
identity fraction: 0.612691
PartField label agreement: 0.929117
```

This setting deliberately keeps much of the proven topology-matched signal while
allowing semantic correction where PartField is confident.
