# Topomatch + Semantic Region Moments Recipe

Date: 2026-06-22

## Why the semantic-vcorr run failed

The previous semantic vertex-correspondence run used PartField features and labels
to choose one target vertex for every source vertex, then optimized the source
vertices directly toward those targets. That is too sharp for these animal pairs.
Even if most correspondences look reasonable statistically, a small bad cluster
or a many-to-one match can pull individual neighborhoods into spikes.

For example, the bulldog PartField labels include tiny buckets:

- bucket 2: 3 vertices
- bucket 11: 14 vertices

Those are not stable enough to act as dense point targets. They are especially
dangerous with a high vertex loss, because a few selected target vertices can
pull a local patch far away from its neighbors.

## New recipe

The new run keeps the best geometric ingredient from `topomatch_vcorr`:

- `TARGET_VERTEX_CORRESPONDENCE_WEIGHT=20000`
- `DEFORMATION_PARAMETERIZATION=vertex`
- `GLOBAL_CHAMFER_WEIGHT_OVERRIDE=0`
- `TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=0`

Then it adds semantics only as low-frequency region statistics from aligned
PartField buckets:

- hard PartField Chamfer: disabled
- soft PartField Chamfer: disabled
- fixed semantic anchors: disabled
- bucket containment: disabled
- bucket centroid moment: enabled
- bucket RMS extent moment: enabled
- bucket trimmed coordinate profile: enabled
- tiny buckets skipped with `PARTFIELD_MIN_POINTS=48`

This means a semantic region such as a leg/head/body bucket is encouraged to
match the target region's center, spread, and coordinate distribution. No source
vertex is told that it must move to one specific semantic target vertex.

## Default job settings

```bash
TARGET_VERTEX_CORRESPONDENCE_WEIGHT=20000.0
TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=0.0
GLOBAL_CHAMFER_WEIGHT_OVERRIDE=0.0
PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=1000.0
PARTFIELD_LABELS_ALIGNED=1
PARTFIELD_MIN_POINTS=48
PARTFIELD_HARD_WEIGHT=0.0
PARTFIELD_SOFT_WEIGHT=0.0
PARTFIELD_MOMENT_WEIGHT=1.0
PARTFIELD_MOMENT_EXTENT_WEIGHT=0.45
PARTFIELD_PROFILE_WEIGHT=0.25
PARTFIELD_PROFILE_BINS=9
PARTFIELD_PROFILE_TRIM=0.10
PARTFIELD_ANCHOR_WEIGHT=0.0
PARTFIELD_CONTAINMENT_WEIGHT=0.0
EDGE_STRETCH_WEIGHT=100.0
EDGE_STRETCH_THRESHOLD=1.35
EDGE_DISPLACEMENT_JUMP_WEIGHT=250.0
EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.25
```

## Local sanity check

A CPU-side loss construction check was run for `cat_to_bulldog`.

Result:

- active buckets: 8
- raw region-stat loss: `0.140731`
- moment raw: `0.129475`
- profile raw: `0.045026`
- hard bucket raw: `0.0`
- anchor raw: `0.0`

Skipped buckets included the unstable small bulldog target buckets.

## Jobs

Single pair:

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_topomatch_semantic_moments_single.sh 5
```

Requested bulldog/cat and bulldog/dachshund directions:

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_topomatch_semantic_moments_array.sh
```

Outputs go to:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/topomatch_semantic_moments/
```
