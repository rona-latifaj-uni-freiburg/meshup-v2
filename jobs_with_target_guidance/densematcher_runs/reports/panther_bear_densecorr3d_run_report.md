# DenseCorr3D Panther/Bear Runs

Date: 2026-06-29

## Assumption

The request named the panther explicitly as `071b8_toy_animals_017`. The bear path was not written separately, so I used the other explicitly provided mesh, `bdfd0_toy_animals_016`, as the bear candidate.

## DenseCorr3D Segmentation

DenseCorr3D provides supervised semantic groups in each object's `groups.txt` on `simple_mesh.obj`. Each line is one semantic bucket, and same-category objects use the same line order. These runs use those group ids as hard semantic part buckets, not PartField clustering.

Both meshes have 8 buckets.

### Original Mesh Counts

| object | color mesh vertices | color mesh faces | simple mesh vertices | buckets |
| --- | ---: | ---: | ---: | ---: |
| `071b8_toy_animals_017` panther | 145127 | 290250 | 1656 | 8 |
| `bdfd0_toy_animals_016` bear candidate | 217836 | 435680 | 1801 | 8 |

## Prepared Variants

Prepared directory:

`jobs_with_target_guidance/densematcher_runs/prepared/panther_bear`

Segmentation preview:

`jobs_with_target_guidance/densematcher_runs/reports/071b8_panther_bdfd0_bear_densecorr3d_segmentation_views.png`

Final/target overlay preview:

`jobs_with_target_guidance/densematcher_runs/reports/panther_bear_densecorr3d_final_target_overlay_views.png`

### `fullmatch145127`

The panther was kept at full resolution. The larger bear candidate was decimated to the same vertex count while transferring DenseCorr3D labels from the full labeled mesh.

| object | vertices | faces |
| --- | ---: | ---: |
| `071b8_toy_animals_017` | 145127 | 290250 |
| `bdfd0_toy_animals_016` | 145127 | 290262 |

### `equal4996`

Both meshes were made exactly 4996 vertices while preserving the semantic bucket labels.

| object | vertices | faces |
| --- | ---: | ---: |
| `071b8_toy_animals_017` | 4996 | 9988 |
| `bdfd0_toy_animals_016` | 4996 | 10000 |

## Run Recipe

All four jobs used the dog-run-style settings:

- `EPOCHS=4000`
- `PARTFIELD_USE_FEATURES=0`
- `PARTFIELD_LABELS_ALIGNED=1`
- `PARTFIELD_N_BUCKETS=8`
- `PARTFIELD_GUIDANCE_MODE=hard`
- `PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=8000.0`
- `GLOBAL_CHAMFER_WEIGHT_OVERRIDE=750.0`
- `PARTFIELD_SOURCE_TO_TARGET_WEIGHT=1.0`
- `PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0`
- `PARTFIELD_MIN_POINTS=12`
- `JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000.0`
- `EDGE_DISPLACEMENT_JUMP_WEIGHT=500.0`
- `ENABLE_SOURCE_DINO_LOSS=0`
- `ENABLE_TARGET_DINO_GUIDANCE=0`

## Completed Jobs

| job | direction | variant | state | elapsed | output |
| ---: | --- | --- | --- | ---: | --- |
| 5392134 | panther -> bear | `fullmatch145127` | completed | 00:08:28 | `jobs_with_target_guidance/densematcher_runs/outputs/panther_bear/071b8_toy_animals_017_to_bdfd0_toy_animals_016_fullmatch145127_dev_h100_dogrecipe_global750_hard_partfield_chamfer_only_densecorr3d_fullmatch145127_groups_4000ep_5392134` |
| 5392135 | bear -> panther | `fullmatch145127` | completed | 00:07:28 | `jobs_with_target_guidance/densematcher_runs/outputs/panther_bear/bdfd0_toy_animals_016_to_071b8_toy_animals_017_fullmatch145127_dev_h100_dogrecipe_global750_hard_partfield_chamfer_only_densecorr3d_fullmatch145127_groups_4000ep_5392135` |
| 5392136 | bear -> panther | `equal4996` | completed | 00:04:10 | `jobs_with_target_guidance/densematcher_runs/outputs/panther_bear/bdfd0_toy_animals_016_to_071b8_toy_animals_017_equal4996_dev_h100_dogrecipe_global750_hard_partfield_chamfer_only_densecorr3d_equal4996_groups_4000ep_5392136` |
| 5392137 | panther -> bear | `equal4996` | completed | 00:04:02 | `jobs_with_target_guidance/densematcher_runs/outputs/panther_bear/071b8_toy_animals_017_to_bdfd0_toy_animals_016_equal4996_dev_h100_dogrecipe_global750_hard_partfield_chamfer_only_densecorr3d_equal4996_groups_4000ep_5392137` |

Each output directory contains `mesh_final/mesh.obj`, `colored_meshes/mesh_final_position.ply`, and `correspondence/final_correspondence.json`.
