# DenseCorr3D Bucket-Balanced Requested Animal Runs

Submitted: 2026-07-02

## Request

Run the cat-to-dachshund `best_asym035` hard partfield-chamfer recipe on
DenseCorr3D animal meshes, with no mesh above 6-7k vertices and with matching
semantic bucket counts within each source/target pair.

## Recipe

- `EPOCHS=4000`
- `PARTFIELD_USE_FEATURES=0`
- `PARTFIELD_N_BUCKETS=8`
- `PARTFIELD_LABELS_ALIGNED=1`
- `PARTFIELD_CHAMFER_WEIGHT=8000`
- `PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.35`
- `PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0`
- `JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000`
- `EDGE_DISPLACEMENT_JUMP_WEIGHT=500`
- `EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.2`
- `PARTFIELD_GUIDANCE_MODE=hard`

## Balanced 5k Variants

Prepared directory:

`jobs_with_target_guidance/densematcher_runs/prepared/requested_animals_20260630`

| Pair bucket | Variant | Vertices per mesh | Shared bucket counts |
| --- | --- | ---: | --- |
| panther / cheetah | `panther_cheetah_bucketavg4999` | 4999 | `[410, 2359, 418, 471, 483, 202, 589, 67]` |
| bear / panther | `bear_panther_bucketavg5002` | 5002 | `[458, 2260, 421, 403, 558, 210, 600, 92]` |
| bear / elephant | `bear_elephant_bucketavg5001` | 5001 | `[439, 1998, 402, 399, 639, 354, 438, 332]` |
| elephant / moose | `elephant_moose_bucketavg5001` | 5001 | `[367, 1604, 346, 367, 617, 391, 327, 982]` |
| elephant / giraffe | `elephant_giraffe_bucketavg4998` | 4998 | `[444, 1747, 460, 375, 875, 348, 420, 329]` |

Each variant was generated with
`jobs_with_target_guidance/densecorr3d_balance_bucket_counts.py` from the
existing `5k` prepared meshes, using `--target-mode average --max-vertices 6000`.
Mesh vertex counts were validated against the label arrays after generation.

## Chained Slurm Submission

Submit script:

`jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_bucketbalanced_requested_single.sh`

Initial job:

`5698613`

The first submission attempt was rejected by `QOSMaxSubmitJobPerUserLimit`
because existing `sam3d-recon-5k` / `sam2-mask-adapt` jobs occupied the user
limit. The second attempt was accepted and is pending behind those jobs.

Output root:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_bucketbalanced_requested_5k_best_asym035`

Chain report:

`jobs_with_target_guidance/cross_animal_spike_runs/reports/densecorr3d_bucketbalanced_requested_chain.txt`

Task order:

| Task | Deformation | Pair id | Variant |
| ---: | --- | ---: | --- |
| 0 | panther -> cheetah | 8 | `panther_cheetah_bucketavg4999` |
| 1 | bear -> panther | 5 | `bear_panther_bucketavg5002` |
| 2 | cheetah -> panther | 9 | `panther_cheetah_bucketavg4999` |
| 3 | bear -> elephant | 10 | `bear_elephant_bucketavg5001` |
| 4 | elephant -> moose | 7 | `elephant_moose_bucketavg5001` |
| 5 | elephant -> giraffe | 1 | `elephant_giraffe_bucketavg4998` |
| 6 | giraffe -> elephant | 0 | `elephant_giraffe_bucketavg4998` |
