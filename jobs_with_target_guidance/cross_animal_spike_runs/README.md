# Cross-animal Spike Runs

This folder is for the follow-up animal deformation sweep requested on 2026-06-20.

## Layout

- `jobs/`: SLURM wrappers and pair runner.
- `logs/`: SLURM stdout/stderr.
- `partfield/features/no_dino_animals/`: fresh PartField feature `.npy` files.
- `partfield/segments/no_dino_animals_12/`: fresh 12-bucket aligned labels and colored PartField meshes.
- `outputs/topomatch_vcorr/`: current corrected recipe for these same-topology animal morph outputs.
- `outputs/topomatch_semantic_moments/`: shape-first recipe that keeps topology-matched vertex guidance and adds only low-frequency PartField semantic region moments/profiles.
- `outputs/topomatch_semantic_reindexed/`: no-geometry-change semantic correspondence pass; reindexes clean topomatch meshes by a bijective semantic vertex assignment.
- `outputs/semantic_vcorr_partfield/`: semantic dense-correspondence bridge using PartField features.
- `correspondences/`: cached source-vertex to target-vertex semantic maps.
- `outputs/topomatch_vcorr_dense_pca/`: 250-epoch rerun with dense epoch renders and DINO PCA snapshots.
- `outputs/momenthard_asym040_shape/`: previous target-shape recipe with semantic bucket moment matching.
- `outputs/bodyfix_hard_asym030_jump1200/`: body-preserving anti-spike recipe; no spikes, but too source-like.
- `outputs/gatedhybrid_asym035_jump500/`: previous over-regularized recipe; no spikes, but shrunken/scrunched bodies.
- `outputs/best_asym035_jump500/`: earlier hard-bucket baseline outputs, when present.
- `reports/`: simple per-run and aggregate reports.

## Deformation pairs

The array job runs these directions:

| Pair ID | Direction |
|---:|---|
| 0 | dachshund to golden retriever |
| 1 | golden retriever to dachshund |
| 2 | dachshund to cat |
| 3 | cat to dachshund |
| 4 | bulldog to cat |
| 5 | cat to bulldog |
| 6 | bulldog to dachshund |
| 7 | dachshund to bulldog |

## Recipe

The current recipe uses direct vertex-to-vertex target guidance when source and
target have identical topology. These cross-animal dog-morph meshes all share
the same vertex count and face indices, so vertex identity is the strongest
semantic correspondence available and avoids Chamfer nearest-neighbor spikes.

- deformation parameterization: `vertex`
- global Chamfer weight: `0`
- PartField Chamfer weight: `0`
- topology-matched target vertex correspondence weight: `20000`
- topology check: source/target vertex counts and face tensors must match
- SDS, source DINO, and target-render guidance disabled

The older PartField/Chamfer recipes remain useful for genuinely different
topologies, but they are the wrong tool for this specific cross-animal folder.

## Semantic Vertex-Correspondence Bridge

For a topology-independent variant, use the new semantic vertex-correspondence
loss. It builds a dense source-vertex to target-vertex map from PartField
features, PartField labels, normalized position, and normals, then runs the same
direct vertex target loss against those semantic target indices.

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_semantic_vcorr_array.sh
```

The default bridge disables identity vertex loss, global Chamfer, and PartField
Chamfer, then uses:

- semantic vertex-correspondence weight: `20000`
- soft PartField label filtering
- topology prior: `0.20`, so same-topology runs keep identity matches unless
  PartField gives a better semantic target

Details and the local dachshund-to-cat correspondence check are in
`reports/semantic_vertex_correspondence_bridge_2026-06-21.md`.

## Topomatch + Semantic Region Moments

The semantic vertex-correspondence bridge produced spike artifacts because it
turned noisy semantic matches into direct per-vertex targets. The safer follow-up
keeps `topomatch_vcorr` as the main body-preserving constraint and uses
PartField semantics only as region-level moments/profiles:

- direct topology-matched vertex loss: enabled, weight `20000`
- semantic vertex correspondence loss: disabled
- global Chamfer: disabled
- hard/soft PartField Chamfer: disabled
- PartField anchors: disabled
- PartField bucket centroid/RMS/profile matching: enabled
- tiny PartField buckets skipped with `PARTFIELD_MIN_POINTS=48`

Submit the requested cat/bulldog and dachshund/bulldog directions with:

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_topomatch_semantic_moments_array.sh
```

Details are in `reports/topomatch_semantic_moments_recipe_2026-06-22.md`.

## Semantic Reindexing

The latest safer path does not use semantics as a deformation force. It keeps
the best topomatch geometry and computes a post-hoc semantic vertex permutation.
The reindexed OBJ is verified to recover the input topomatch mesh exactly under
the inverse permutation, so this pass cannot introduce spikes or waves.

Details and output locations are in
`reports/semantic_reindex_topomatch_2026-06-23.md`.

## Dense/PCA inspection run

The dense inspection run saves normal render grids at epochs `1..50` and then
`60,70,...,250`. It also saves DINO PCA grids at epoch `1` and every 10 epochs.
PCA files are written under each run as
`pca_visualization/epoch_XXXXX/grid_pca_all_views.png`; normal mesh renders are
written as `epoch_renders/epoch_XXXXX/grid_all_views.png`.

## Submit manually

```bash
pf_job=$(sbatch --parsable jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_prepare_partfield_cross_animals.sh)
def_job=$(sbatch --parsable --dependency=afterok:${pf_job} --array=0-7 jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_best_array.sh)
sbatch --dependency=afterok:${def_job} jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_analyze_cross_animal_outputs.sh
```
