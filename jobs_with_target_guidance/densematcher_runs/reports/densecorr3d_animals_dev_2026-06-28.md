# DenseCorr3D Animals Dev Run

Date: 2026-06-28

## What Changed

- Added `jobs_with_target_guidance/densecorr3d_segment.py` to convert DenseCorr3D
  `groups.txt` annotations into MeshUp-compatible vertex-label NPZs.
- Added DenseCorr3D animal run wrappers under
  `jobs_with_target_guidance/densematcher_runs/jobs/`.
- Extended the existing Artur Chamfer runner with `PARTFIELD_USE_FEATURES=0`
  so supervised label-only guidance does not require PartField feature arrays.

## Data

- Source: `external/DenseCorr3D/animals/071b8_toy_animals_017/simple_mesh.obj`
- Target: `external/DenseCorr3D/animals/13cf7_toy_animals_055/simple_mesh.obj`
- Labels:
  `jobs_with_target_guidance/densematcher_runs/densecorr3d/animals/labels/`
- DenseCorr3D animal subset size after extraction: about `369M`.
- Download archive was removed after extraction.

DenseCorr3D groups converted as 8 aligned buckets:

| bucket | source vertices | target vertices |
|---:|---:|---:|
| 0 | 140 | 99 |
| 1 | 766 | 668 |
| 2 | 151 | 107 |
| 3 | 112 | 131 |
| 4 | 147 | 253 |
| 5 | 102 | 78 |
| 6 | 218 | 155 |
| 7 | 20 | 127 |

## Run

Submitted Slurm job `5278078`:

```bash
DENSECORR3D_ROOT=external/DenseCorr3D \
sbatch jobs_with_target_guidance/densematcher_runs/jobs/job_dev_h100_densecorr3d_animal_pair.sh \
  071b8_toy_animals_017 \
  13cf7_toy_animals_055
```

Output:

```text
jobs_with_target_guidance/densematcher_runs/outputs/animals_dev/071b8_toy_animals_017_to_13cf7_toy_animals_055_dev_h100_hard_partfield_chamfer_only_densecorr3d_groups_300ep_5278078/
```

The job completed successfully in about 1 minute 51 seconds. The run used:

- hard bucket Chamfer only
- `PARTFIELD_USE_FEATURES=0`
- `PARTFIELD_LABELS_ALIGNED=1`
- `target_mesh_partfield_chamfer_weight=8000`
- `partfield_source_to_target_weight=0.35`
- `partfield_target_to_source_weight=1.0`
- jacobian parameterization with neighbor smoothness and edge-displacement jump guard

## Quick Read

The integration works end-to-end: DenseCorr3D labels were consumed as aligned
hard buckets, all 8 buckets were active, and final mesh/render outputs were
produced. The visual result is not yet high quality; the 300-epoch hard Chamfer
recipe folds parts of the body. The next step should tune the recipe, likely
with region moments/profiles or a gentler staged schedule, now that supervised
DenseCorr3D labels are available.
