# DenseCorr3D 2d6b3 Elephant -> 34fb4 Giraffe Debug

Date: 2026-06-28

## Corrected Pair

- Source elephant: `external/DenseCorr3D/animals/2d6b3_toy_animals_009/color_mesh.obj`
- Target giraffe: `external/DenseCorr3D/animals/34fb4_toy_animals_019/color_mesh.obj`

Prepared directory:

```text
jobs_with_target_guidance/densematcher_runs/prepared/elephant2d6b3_giraffe
```

## Segmentation Visualization

PNG visualization:

```text
jobs_with_target_guidance/densematcher_runs/reports/2d6b3_to_34fb4_densecorr3d_segmentation_views.png
```

Colored PLY diagnostics:

```text
jobs_with_target_guidance/densematcher_runs/prepared/elephant2d6b3_giraffe/colored/2d6b3_toy_animals_009_densecorr3d_full_groups.ply
jobs_with_target_guidance/densematcher_runs/prepared/elephant2d6b3_giraffe/colored/34fb4_toy_animals_019_densecorr3d_full_groups.ply
jobs_with_target_guidance/densematcher_runs/prepared/elephant2d6b3_giraffe/colored/2d6b3_toy_animals_009_densecorr3d_equal4996_groups.ply
jobs_with_target_guidance/densematcher_runs/prepared/elephant2d6b3_giraffe/colored/34fb4_toy_animals_019_densecorr3d_equal4996_groups.ply
```

DenseCorr3D animal bucket count: 8.

Full transferred labels:

- `2d6b3` elephant: 190950 vertices, 381900 faces
- `34fb4` giraffe: 130100 vertices, 260208 faces

Default 5k labels:

- `2d6b3` elephant: 5000 vertices, 10000 faces
- `34fb4` giraffe: 4996 vertices, 10000 faces

Equalized labels:

- `2d6b3` elephant: 4996 vertices, 9992 faces
- `34fb4` giraffe: 4996 vertices, 10000 faces

All 8 buckets are nonempty on both equalized meshes.

## Why The Previous Run Was Wrinkly

Previous bad full run:

```text
jobs_with_target_guidance/densematcher_runs/outputs/elephant_to_giraffe/071b8_toy_animals_017_to_34fb4_toy_animals_019_full_h100_densecorr3d_hard_partfield_chamfer_only_densecorr3d_full_groups_4000ep_5278932
```

Main differences from the good bulldog run:

1. Wrong source object for the user-requested pair:
   the run used `071b8_toy_animals_017`, but the requested elephant mesh is
   `2d6b3_toy_animals_009`.

2. The bad DenseCorr3D run disabled global Chamfer:

   ```text
   GLOBAL_CHAMFER_WEIGHT=0.0
   ```

   The good bulldog run used:

   ```text
   GLOBAL_CHAMFER_WEIGHT=750.0
   ```

   With global Chamfer off, the full-resolution mesh is only constrained by
   sampled per-bucket part Chamfer. That is too sparse for 145k+ vertices and
   can leave local freedom that shows up as wrinkles.

3. The bad full run optimized a very high-resolution source:

   ```text
   145127 vertices, 290250 faces
   ```

   The good bulldog run optimized:

   ```text
   4712 vertices, 9420 faces
   ```

   The high-res mesh has many more deformation degrees of freedom while the
   partfield Chamfer still samples only 512 points per bucket.

4. DenseCorr3D labels are transferred from `simple_mesh.obj` to `color_mesh.obj`.
   They are supervised semantic buckets, but on the full mesh they are still a
   nearest-neighbor transfer from the annotated simple mesh. The visualization
   should be inspected before trusting a run.

## Corrected Run

I created an exact equal-vertex-count variant named `equal4996`.

Submitted command:

```bash
sbatch --partition=dev_gpu_h100 --time=00:30:00 --job-name=dm_2d6b3_giraffe_eq --export=ALL,PREPARED_DIR=./jobs_with_target_guidance/densematcher_runs/prepared/elephant2d6b3_giraffe,OUTPUT_ROOT=./jobs_with_target_guidance/densematcher_runs/outputs/elephant2d6b3_to_giraffe,RUN_TAG=dev_h100_eqglobal750,GLOBAL_CHAMFER_WEIGHT_OVERRIDE=750.0,PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.35,PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0,LOG_INTERVAL_IM=250,SAVE_RENDERS_INTERVAL=250 jobs_with_target_guidance/densematcher_runs/jobs/job_h100_densecorr3d_prepared_pair.sh equal4996 2d6b3_toy_animals_009 34fb4_toy_animals_019 4000
```

Job id:

```text
5278965
```

Confirmed from the run log:

- Source mesh: `2d6b3_toy_animals_009_densecorr3d_equal4996.obj`
- Target mesh: `34fb4_toy_animals_019_densecorr3d_equal4996.obj`
- Global Chamfer: `750.0`
- Partfield Chamfer: `8000.0`
- DenseCorr3D buckets: `8`
- Epochs: `4000`

Completion:

```text
State: COMPLETED
ExitCode: 0:0
Elapsed: 00:04:10
Node: uc3n082
```

Final output:

```text
jobs_with_target_guidance/densematcher_runs/outputs/elephant2d6b3_to_giraffe/2d6b3_toy_animals_009_to_34fb4_toy_animals_019_equal4996_dev_h100_eqglobal750_hard_partfield_chamfer_only_densecorr3d_equal4996_groups_4000ep_5278965
```

Final mesh:

```text
jobs_with_target_guidance/densematcher_runs/outputs/elephant2d6b3_to_giraffe/2d6b3_toy_animals_009_to_34fb4_toy_animals_019_equal4996_dev_h100_eqglobal750_hard_partfield_chamfer_only_densecorr3d_equal4996_groups_4000ep_5278965/mesh_final/mesh.obj
```

Initial/final quick visual:

```text
jobs_with_target_guidance/densematcher_runs/reports/2d6b3_to_34fb4_equal4996_run_initial_final_views.png
```
