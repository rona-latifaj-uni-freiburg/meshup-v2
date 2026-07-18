# DenseMatcher / DenseCorr3D Runs

This folder wires the supervised DenseCorr3D annotations from DenseMatcher into
the existing MeshUp target-guidance pipeline.

DenseMatcher's dataset release stores each object as:

- `simple_mesh.obj`
- `groups.txt`
- `groups_visualization.obj`
- `color_mesh.obj`

`groups.txt` is converted into the same label NPZ format consumed by
`partfield_chamfer.py`, then the normal hard bucket Chamfer runner is used with
`PARTFIELD_USE_FEATURES=0`.

## Prepare Labels

Download and unzip DenseCorr3D outside git, then point the converter at it:

```bash
python jobs_with_target_guidance/densecorr3d_segment.py \
  --densecorr3d-root /path/to/DenseCorr3D \
  --category animals \
  --objects 071b8_toy_animals_017 13cf7_toy_animals_055 \
  --output-dir jobs_with_target_guidance/densematcher_runs/densecorr3d/animals
```

The output layout is:

```text
densecorr3d/animals/
  labels/<object>_densecorr3d_labels.npz
  colored/<object>_densecorr3d_groups.ply
  summary.json
```

## Run A Pair

```bash
DENSECORR3D_ROOT=/path/to/DenseCorr3D \
sbatch jobs_with_target_guidance/densematcher_runs/jobs/job_dev_h100_densecorr3d_animal_pair.sh \
  071b8_toy_animals_017 \
  13cf7_toy_animals_055
```

The job auto-runs the label converter for the two objects if the label NPZs do
not exist yet.
