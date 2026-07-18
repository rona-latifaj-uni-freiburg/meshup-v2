# DenseCorr3D Elephant -> Giraffe Run

Date: 2026-06-28

## Dataset Placement

I downloaded the official DenseCorr3D archive linked from the DenseMatcher
repository and extracted it to:

```text
external/DenseCorr3D
```

Verification of the extracted tree:

- Top-level categories: 24
- Object directories with `groups.txt`: 600 in the downloaded archive
- Objects missing any of `color_mesh.obj`, `simple_mesh.obj`, `groups.txt`,
  or `groups_visualization.obj`: 0
- Animal objects: 10
- Animal semantic groups per object: 8

The temporary `external/DenseCorr3D.zip` download was removed after extraction
was verified.

## What The DenseCorr3D Segmentation Is

DenseCorr3D does not give us an unsupervised PartField-style clustering. It
gives supervised dense correspondence annotations in `groups.txt`.

For each object:

- `color_mesh.obj` is the original colored mesh.
- `simple_mesh.obj` is a remeshed/simplified mesh, around 2000 vertices.
- `groups.txt` contains the supervised semantic correspondence groups.
- `groups_visualization.obj` is only a colored visualization of those groups.

Each non-empty line in `groups.txt` is one semantic group. The integers on that
line are vertex indices on `simple_mesh.obj`. For objects in the same category,
line 0 corresponds to line 0, line 1 to line 1, and so on. For the animal
category, that gives 8 aligned semantic buckets.

For the full-resolution runs I transferred those supervised simple-mesh labels
onto `color_mesh.obj` by nearest neighbor in per-object bbox-normalized
coordinates. For the 5k runs I decimated the full mesh with pymeshlab, then
transferred the full-mesh labels onto the decimated vertices.

References:

- https://github.com/TEA-Lab/DenseMatcher
- https://tea-lab.github.io/DenseMatcher/
- https://arxiv.org/abs/2412.05268

## Code Added

- `jobs_with_target_guidance/densecorr3d_segment.py`
  Converts DenseCorr3D `groups.txt` files on `simple_mesh.obj` into MeshUp
  label NPZs and colored PLY diagnostics.

- `jobs_with_target_guidance/densecorr3d_prepare_mesh_variants.py`
  Transfers DenseCorr3D supervised buckets to `color_mesh.obj` and prepares
  `full` and `5k` mesh variants with matching NPZ labels.

- `jobs_with_target_guidance/densematcher_runs/jobs/job_h100_densecorr3d_prepared_pair.sh`
  Runs the existing hard part-Chamfer pipeline on prepared DenseCorr3D mesh
  variants.

The existing `run_artur_chamfer_ablation.sh` path was used with:

- `PARTFIELD_USE_FEATURES=0`
- `PARTFIELD_LABELS_ALIGNED=1`
- `PARTFIELD_N_BUCKETS=8`
- `PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=8000.0`
- `PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.35`
- `PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0`
- `DEFORMATION_PARAMETERIZATION=jacobian`
- `JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000.0`
- `EDGE_DISPLACEMENT_JUMP_WEIGHT=500.0`

## Pair Selection

I rendered the animal category in:

```text
jobs_with_target_guidance/densematcher_runs/reports/animals_color_multiview.png
```

Selected IDs:

- Elephant source: `071b8_toy_animals_017`
- Giraffe target: `34fb4_toy_animals_019`

## Prepared Meshes

Prepared directory:

```text
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe
```

Elephant:

- Source simple annotation mesh: 1656 vertices
- Full prepared mesh: 145127 vertices, 290250 faces
- 5k prepared mesh: 5002 vertices, 10000 faces
- All 8 buckets are nonempty in both variants.

Giraffe:

- Target simple annotation mesh: 1896 vertices
- Full prepared mesh: 130100 vertices, 260208 faces
- 5k prepared mesh: 4996 vertices, 10000 faces
- All 8 buckets are nonempty in both variants.

Primary files:

```text
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/meshes/071b8_toy_animals_017_densecorr3d_full.obj
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/meshes/34fb4_toy_animals_019_densecorr3d_full.obj
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/labels/071b8_toy_animals_017_densecorr3d_full_labels.npz
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/labels/34fb4_toy_animals_019_densecorr3d_full_labels.npz
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/meshes/071b8_toy_animals_017_densecorr3d_5k.obj
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/meshes/34fb4_toy_animals_019_densecorr3d_5k.obj
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/labels/071b8_toy_animals_017_densecorr3d_5k_labels.npz
jobs_with_target_guidance/densematcher_runs/prepared/elephant_giraffe/labels/34fb4_toy_animals_019_densecorr3d_5k_labels.npz
```

## Submitted Jobs

Commands submitted:

```bash
sbatch jobs_with_target_guidance/densematcher_runs/jobs/job_h100_densecorr3d_prepared_pair.sh full 071b8_toy_animals_017 34fb4_toy_animals_019 4000
sbatch jobs_with_target_guidance/densematcher_runs/jobs/job_h100_densecorr3d_prepared_pair.sh 5k 071b8_toy_animals_017 34fb4_toy_animals_019 4000
```

Initial Slurm jobs:

- Full-resolution run: `5278928`
- 5k run: `5278927`

Both jobs were submitted to `gpu_h100` with a 2:30:00 walltime and 4000 epochs.
Slurm cancelled both before allocation with reason `Priority`, elapsed
`00:00:00`, and no node assigned.

Fallback dev Slurm jobs:

```bash
sbatch --partition=dev_gpu_h100 --time=00:30:00 --job-name=dm_eg_full4k jobs_with_target_guidance/densematcher_runs/jobs/job_h100_densecorr3d_prepared_pair.sh full 071b8_toy_animals_017 34fb4_toy_animals_019 4000
sbatch --partition=dev_gpu_h100 --time=00:30:00 --job-name=dm_eg_5k4k jobs_with_target_guidance/densematcher_runs/jobs/job_h100_densecorr3d_prepared_pair.sh 5k 071b8_toy_animals_017 34fb4_toy_animals_019 4000
```

- Full-resolution dev run: `5278932`
- 5k dev run: `5278933`

Latest check after resubmission:

- `5278932` full-resolution dev run: pending in `dev_gpu_h100`, reason
  `Resources`
- `5278933` 5k dev run: pending in `dev_gpu_h100`, reason `Priority`

Output root:

```text
jobs_with_target_guidance/densematcher_runs/outputs/elephant_to_giraffe
```

Log root:

```text
jobs_with_target_guidance/densematcher_runs/logs
```
