# Semantic Reindexing on Topomatch Geometry

Date: 2026-06-23

## Reason for the pivot

The semantic-vcorr and semantic-moment losses both damaged geometry. Even
low-frequency semantic statistics could still fight the strong topomatch body
anchor and produce wavy surfaces. The safer solution is to stop using semantic
correspondence as a deformation loss.

This recipe separates the two jobs:

1. Use the clean `topomatch_vcorr` deformation for geometry.
2. Build a semantic source-vertex to target-vertex bijection after the fact.
3. Reindex the already-deformed mesh with that bijection.

The reindexing step is a pure vertex renumbering. It does not move vertices.

## What the script does

Script:

```text
jobs_with_target_guidance/semantic_reindex_topomatch.py
```

Runner:

```text
jobs_with_target_guidance/cross_animal_spike_runs/jobs/run_semantic_reindex_topomatch_pair.sh
```

For equal vertex counts, the script builds a one-to-one semantic permutation:

- descriptor = PartField feature + weak bbox position + weak normal
- nearest candidates are found with `topk=256`
- label mismatches are penalized, but labels are not hard constraints
- assignment is greedy one-to-one, so no target vertex is reused

Then it writes:

- `mesh_semantic_reindexed.obj`
- `semantic_permutation.npz`
- `semantic_index_map.csv`
- `summary.json`

The output OBJ is geometrically identical to the topomatch input up to a pure
renumbering. This is verified by reconstructing the original mesh from the
permutation.

## Completed outputs

### Bulldog to cat

Input topomatch:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_vcorr/bulldog_to_cat_topomatch_vcorr_dev_h100_hard_partfield_chamfer_only_topomatch_vcorr20000_vertex_2500ep_5090991/mesh_final/mesh.obj
```

Output:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_semantic_reindexed/bulldog_to_cat_semantic_reindexed_from_bulldog_to_cat_topomatch_vcorr_dev_h100_hard_partfield_chamfer_only_topomatch_vcorr20000_vertex_2500ep_5090991/
```

Stats:

- identity fraction: `0.090407`
- label agreement: `0.700127`
- mean similarity: `0.552093`
- pure reindex vertex recovery max abs error: `0.0`
- faces recover exactly: `true`

### Cat to bulldog

Input topomatch:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_vcorr/cat_to_bulldog_topomatch_vcorr_dev_h100_hard_partfield_chamfer_only_topomatch_vcorr20000_vertex_2500ep_5090994/mesh_final/mesh.obj
```

Output:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_semantic_reindexed/cat_to_bulldog_semantic_reindexed_from_cat_to_bulldog_topomatch_vcorr_dev_h100_hard_partfield_chamfer_only_topomatch_vcorr20000_vertex_2500ep_5090994/
```

Stats:

- identity fraction: `0.092954`
- label agreement: `0.689092`
- mean similarity: `0.549115`
- pure reindex vertex recovery max abs error: `0.0`
- faces recover exactly: `true`

### Bulldog to dachshund

Only a 250-epoch clean topomatch inspection mesh was present locally. Full
2500-epoch topomatch was queued as job `5167154`.

Output from available 250-epoch topomatch:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_semantic_reindexed/bulldog_to_dachshund_semantic_reindexed_from_bulldog_to_dachshund_topomatch_vcorr_densepca_h100_hard_partfield_chamfer_only_topomatch_vcorr20000_vertex_250ep_5108401/
```

Stats:

- identity fraction: `0.162564`
- label agreement: `0.858234`
- mean similarity: `0.709023`
- pure reindex vertex recovery max abs error: `0.0`
- faces recover exactly: `true`

### Dachshund to bulldog

Only a 250-epoch clean topomatch inspection mesh was present locally. Full
2500-epoch topomatch was queued as job `5167155`.

Output from available 250-epoch topomatch:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_semantic_reindexed/dachshund_to_bulldog_semantic_reindexed_from_dachshund_to_bulldog_topomatch_vcorr_densepca_h100_hard_partfield_chamfer_only_topomatch_vcorr20000_vertex_250ep_5108406/
```

Stats:

- identity fraction: `0.185696`
- label agreement: `0.847199`
- mean similarity: `0.709116`
- pure reindex vertex recovery max abs error: `0.0`
- faces recover exactly: `true`

## Important limitation

This is a semantic indexing/correspondence solution, not a new deformation
force. It gives the geometry quality of topomatch and attaches a semantic
target-index map. If source and target have different vertex counts, this exact
OBJ reindexing cannot be bijective; then the right output is a correspondence
map or a shared remeshed/template domain.
