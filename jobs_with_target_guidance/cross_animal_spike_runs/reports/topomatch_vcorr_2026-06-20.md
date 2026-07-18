# Topology-Matched Vertex Correspondence Fix

Date: 2026-06-20

## Root Cause

The cross-animal inputs in `experiments/dog_morphs/outputs/*/mesh_final/mesh.obj`
share identical vertex counts and face tensors. The previous Chamfer and
PartField nearest-neighbor losses were trying to rediscover correspondences
that already existed by vertex identity, which caused either spikes or
source-like details.

The corrected recipe enables direct target vertex correspondence when topology
matches:

- `deformation_parameterization=vertex`
- `target_mesh_vertex_correspondence_weight=20000`
- `target_mesh_chamfer_weight=0`
- `target_mesh_partfield_chamfer_weight=0`

The training loop checks that source and target vertex counts and face tensors
match before enabling the loss.

## Runs

Output root:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_vcorr`

Chained sweep jobs:

- pair 0 `dachshund_to_golden_retriever`: `5090983`
- pair 1 `golden_retriever_to_dachshund`: `5090984`
- pair 2 `dachshund_to_cat`: `5090987`
- pair 3 `cat_to_dachshund`: `5090988`
- pair 4 `bulldog_to_cat`: `5090991`
- pair 5 `cat_to_bulldog`: `5090994`
- analysis: `5090996`

There is also an earlier proof run for `dachshund_to_cat`: `5090979`.

## Headline Metrics

| Run | Chamfer L2 | Hausdorff L2 | F-score 0.05 | Normal |
|---|---:|---:|---:|---:|
| bulldog_to_cat `5090991` | 0.030182 | 0.050131 | 0.999833 | 0.945696 |
| cat_to_bulldog `5090994` | 0.041791 | 0.071130 | 0.990165 | 0.942838 |
| cat_to_dachshund `5090988` | 0.034487 | 0.055746 | 0.998833 | 0.941844 |
| dachshund_to_cat `5090987` | 0.028995 | 0.049169 | 1.000000 | 0.945521 |
| dachshund_to_golden_retriever `5090983` | 0.037380 | 0.063612 | 0.996000 | 0.949795 |
| golden_retriever_to_dachshund `5090984` | 0.034742 | 0.054127 | 0.999333 | 0.940624 |

