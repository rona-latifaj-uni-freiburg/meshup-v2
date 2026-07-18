# Giraffe To Elephant 5k Repair Notes

Bad run:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_requested_animals_best_asym035/giraffe_to_elephant_5k_densecorr3d_requested_animals_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5546334`

Observed user issues:

- elephant trunk too wide
- tail swirl
- apparent open space in the head

## Diagnosis

The final mesh has the same closed topology as the source:

- source boundary edges: `0`
- target boundary edges: `0`
- final boundary edges: `0`
- nonmanifold edges: `0`

So the head issue is a deformation/folding artifact, not a literal topological hole in the OBJ.

The bad run used only hard bucket Chamfer:

- `PARTFIELD_CHAMFER_WEIGHT=8000`
- `PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.35`
- `PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0`
- `PARTFIELD_MOMENT_WEIGHT=0`
- `PARTFIELD_PROFILE_WEIGHT=0`
- `PARTFIELD_ANCHOR_WEIGHT=0`
- `PARTFIELD_CONTAINMENT_WEIGHT=0`
- `target_mesh_chamfer_weight=0`

That lets a coarse semantic bucket satisfy nearest-neighbor distances while becoming visually wrong.

The worst bucket-size mismatches are:

| Bucket | Source vertices | Target vertices | Source extent | Target extent | Final extent |
| ---: | ---: | ---: | --- | --- | --- |
| 5 | 87 | 609 | `[0.0229, 0.0205, 0.0174]` | `[0.0762, 0.0738, 0.0521]` | `[0.1271, 0.1278, 0.0820]` |
| 7 | 106 | 553 | `[0.0155, 0.0161, 0.0410]` | `[0.0515, 0.0518, 0.0882]` | `[0.0756, 0.0853, 0.1531]` |

This is consistent with over-expansion of small source patches, especially under a strong target-to-source Chamfer term.

## Repair Presets Added

Scripts changed:

- `jobs_with_target_guidance/cross_animal_spike_runs/jobs/run_densecorr3d_requested_animal_pair.sh`
- `jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_ge5k_repair_single.sh`

### First choice: `ge5k_profile_robust`

This keeps hard DenseCorr3D labels but makes the bucket objective shape-aware:

- `target_to_source_weight=0.60`
- `source_to_target_weight=0.45`
- `tgt_to_src_robust_scale=0.05`
- `global_chamfer_weight=75`
- `partfield_chamfer_weight=6000`
- `moment_weight=0.20`
- `moment_extent_weight=1.00`
- `profile_weight=0.35`
- `anchor_weight=0.25`
- `containment_weight=0.15`
- `jacobian_neighbor_smooth_weight=1500`
- `edge_displacement_jump_weight=1000`
- `edge_displacement_jump_threshold=0.9`

Submit:

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_ge5k_repair_single.sh ge5k_profile_robust
```

### Second choice: `ge5k_balanced_profile`

This changes the bucket matching to balanced transport and keeps the shape stabilizers:

- `partfield_guidance_mode=balanced`
- `global_chamfer_weight=100`
- `partfield_chamfer_weight=5000`
- `moment_weight=0.18`
- `moment_extent_weight=1.00`
- `profile_weight=0.40`
- `anchor_weight=0.20`
- `containment_weight=0.20`
- `balanced_sinkhorn_iters=50`
- `jacobian_neighbor_smooth_weight=1500`
- `edge_displacement_jump_weight=1000`
- `edge_displacement_jump_threshold=0.9`

Submit:

```bash
sbatch jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_ge5k_repair_single.sh ge5k_balanced_profile
```

## Recommended Order

Run `ge5k_profile_robust` first. It is closest to the previous successful recipe but should reduce trunk over-inflation and tail/head folding.

Run `ge5k_balanced_profile` only if the first repair still over-expands the trunk, because balanced transport changes the matching behavior more strongly.
