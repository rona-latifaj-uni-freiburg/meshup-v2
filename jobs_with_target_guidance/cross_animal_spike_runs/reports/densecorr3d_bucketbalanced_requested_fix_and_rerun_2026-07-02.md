# DenseCorr3D Bucket-Balanced Requested Animal Runs: Fix and Rerun

Date: 2026-07-02

## Root cause

`jobs_with_target_guidance/densecorr3d_balance_bucket_counts.py`'s edge-collapse,
edge-split, and Laplacian smoothing operations (used to match per-bucket
vertex counts between a source/target animal mesh pair) could each
independently create near-zero-area sliver faces and/or coincident vertex
pairs. This silently corrupted the bear, cheetah, moose, and giraffe sides of
all 5 bucket-balanced mesh variant pairs used in this chain (panther's mesh
happened to come out clean in every pairing it appeared in - dumb luck, not a
design property).

The corruption made the affected source mesh's Poisson/Laplacian linear
system (`NeuralJacobianFields/PoissonSystem.py`) singular, causing
`loop_tracked.py`'s jacobian-to-vertex solve to emit a NaN assertion failure
after just the first optimizer step of training - well before any visible
mesh-quality issue would otherwise show up.

This is also why gradient clipping (the first hypothesis tried) would not
have fixed it: Adam's first-step update magnitude is inherently bounded by
the learning rate regardless of gradient scale, so a single finite-but-large
gradient step could not explain an instant NaN. A genuinely singular linear
system could and did.

## Fix

Three degeneracy guards were added to `densecorr3d_balance_bucket_counts.py`:

- `select_split_edge`: reject a candidate split whose midpoint would coincide
  with an existing vertex, or whose resulting sub-triangles would be
  near-zero area.
- `select_collapse_edge`: reject a candidate collapse if moving the kept
  vertex to the midpoint would make any surviving neighboring face
  near-degenerate.
- `smooth_by_label`: after each Jacobi-style smoothing step, iteratively roll
  back any vertex touching a now-degenerate face to its pre-step position,
  since simultaneous neighbor movement can jointly degenerate a shared face
  even when each vertex's move looks safe checked in isolation.

All three guards use a 1e-9 minimum-triangle-area threshold and account for
`write_obj`'s actual 8-decimal-place OBJ serialization precision
(near-duplicate vertices still distinct in float64 memory can become
byte-identical once written to disk). `remesh_to_counts`'s final validation
now also hard-asserts no coincident vertices / degenerate faces before
saving, so this class of bug will fail loudly at generation time in the
future instead of silently producing a corrupt mesh that only reveals itself
deep into a 4000-epoch training run.

## Mesh variants regenerated and reverified clean

`bear_panther_bucketavg5002`, `panther_cheetah_bucketavg4999`,
`bear_elephant_bucketavg5001`, `elephant_moose_bucketavg5001`,
`elephant_giraffe_bucketavg4998` (all 5 pairs, 10 mesh files total,
independently verified via numpy: 0 duplicate vertices, 0 faces below 1e-9
area in every file).

## Final outcome (job IDs / sacct states)

| Task | Deformation | Job ID | State | Notes |
| ---: | --- | ---: | --- | --- |
| 0 | panther -> cheetah | 5698613 | trained successfully (mesh_final produced, all 4000 epochs); sacct shows FAILED | chain's own end-of-job sbatch call hit `QOSMaxSubmitJobPerUserLimit`, unrelated to training |
| 1 | bear -> panther | 5708092 | COMPLETED | retried with fixed `bear_panther_bucketavg5002` mesh; original attempt 5700256 failed with NaN Poisson solve |
| 2 | cheetah -> panther | 5708134 | COMPLETED | retried with fixed `panther_cheetah_bucketavg4999` mesh; original attempt 5700506 failed with NaN Poisson solve |
| 3 | bear -> elephant | 5708345 | COMPLETED | retried with fixed `bear_elephant_bucketavg5001` mesh; first attempt 5708167 failed with NaN Poisson solve |
| 4 | elephant -> moose | 5708368 | COMPLETED | ran with fixed `elephant_moose_bucketavg5001` mesh |
| 5 | elephant -> giraffe | 5708825 | COMPLETED | ran with fixed `elephant_giraffe_bucketavg4998` mesh |
| 6 | giraffe -> elephant | 5709026 | COMPLETED | ran with fixed `elephant_giraffe_bucketavg4998` mesh |

All 7 chain tasks are COMPLETED. Output meshes are under
`jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_bucketbalanced_requested_5k_best_asym035/`.

The originally-failed output directories (`bear_to_panther...5700256`,
`cheetah_to_panther...5700506`, `bear_to_elephant...5708167`) were left in
place as historical failed runs, all superseded by the new completed job IDs
above.

## Queue notes

The dev QOS on partition `dev_gpu_h100` for user `fr_rl187` allows a maximum
of 4 submitted jobs and 1 running job at a time. For most of this rerun the
queue was also shared with an unrelated `sam3d-recon-5k`/`sam2-mask-adapt`
pipeline and later a `densecorr3d_srcstable` pipeline from another process,
so several `sbatch` submissions needed a short retry/backoff before being
accepted, and jobs queued behind those other pipelines' running jobs before
starting.
