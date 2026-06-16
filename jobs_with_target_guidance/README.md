# Part-Aware Target Guidance

This folder contains the next target-guidance direction for MeshUp: make the target pull geometry directly, but keep the pull part-aware so vertex identity stays meaningful.

## What Changed

- `part_aware_chamfer.py` adds a part-aware Chamfer loss.
- `partfield_chamfer.py` adds a PartField-feature bucket Chamfer loss.
- `../main.py` exposes optional target Chamfer flags.
- `../loop_tracked.py` calls each target Chamfer loss only when its weight is greater than zero.

The default car schema uses the SAM3D car orientation:

- `x`: left/right width
- `y`: vertical/up
- `z`: car length

It slices both source and target into rough buckets such as cabin/roof, hood/trunk, side body, bumper, and four lower wheel-zone buckets. Source labels are frozen from the initial mesh, so a source roof vertex remains a roof vertex while it moves.

## Run The Car Experiment

From repo root:

```bash
sbatch jobs_with_target_guidance/jobs/job_dev_blueberry_car_part_chamfer.sh
```

Array task `0` runs:

- source: `blueberry_5k_upright_wheels_down.ply`
- prompt: `a sports car`
- target: `bugatti-centodieci_5k_upright_wheels_down.ply`

Array task `1` runs:

- source: `blueberry_5k_upright_wheels_down.ply`
- prompt: `an SUV`
- target: `santa_fe_5k_upright_wheels_down.ply`

Outputs go to:

```text
jobs_with_target_guidance/outputs/
```

## Useful Knobs

- `target_mesh_part_chamfer_weight`: main strength of part-aware target pull.
- `target_mesh_chamfer_weight`: optional small global Chamfer that keeps the whole silhouette glued together.
- `target_mesh_part_chamfer_points`: vertices sampled per active part.
- `target_mesh_part_chamfer_schema`: `car` for the SAM3D cars, `bbox` for a generic spatial grid.
- `target_mesh_part_chamfer_flip_target_longitudinal`: use this if source and target front/rear are reversed.

The loss logs per-part TensorBoard scalars under:

```text
target_mesh_part_chamfer/
```

## PartField Buckets

PartField is a better fit than the hand-written car slices when you want source and target parts to be semantically meaningful. The upstream project predicts a learned 3D part feature field, then clusters those features into a part hierarchy:

```text
https://github.com/nv-tlabs/PartField
https://arxiv.org/abs/2504.11451
```

The MeshUp integration does not install or run the heavy PartField model. Run PartField in its own environment, then point MeshUp at the exported `part_feat_*_0*.npy` files. MeshUp will:

- load source and target PartField features,
- auto-detect face-level versus vertex-level feature arrays,
- average face features onto vertices when needed,
- cluster source vertices into buckets,
- assign target vertices to the nearest source bucket in the same PartField feature space,
- freeze source labels while the source mesh deforms.

This gives you:

```text
source wheel-ish bucket -> target wheel-ish bucket
source cabin-ish bucket -> target cabin-ish bucket
```

instead of global Chamfer pulling every vertex toward every other vertex.

For a softer version that avoids hard bucket IDs, use PartField features as the
correspondence space:

```yaml
partfield_guidance_mode: soft
partfield_soft_match_space: latent
partfield_soft_weight: 1.0
partfield_soft_points: 1024
partfield_soft_semantic_weight: 1.0
partfield_soft_temperature: 0.05
```

This builds the source-target affinity matrix from normalized PartField latent
features only, then applies the differentiable geometry residual through that
soft semantic correspondence. A purely feature-only loss would be constant
unless PartField is re-run differentiably on the deformed mesh.

### PartField Setup Summary

Use the official PartField repo and checkpoint outside this repo. The README command pattern is:

```bash
python partfield_inference.py \
  -c configs/final/demo.yaml \
  --opts continue_ckpt model/model_objaverse.ckpt \
  result_name partfield_features/car_5k \
  dataset.data_path data/car_5k
```

For mesh clustering, the official repo notes that multi-component or poorly connected meshes can behave better with KNN/MST connectivity:

```bash
python run_part_clustering.py \
  --root exp_results/partfield_features/car_5k \
  --dump_dir exp_results/clustering/car_5k \
  --source_dir data/car_5k \
  --use_agglo True \
  --max_num_clusters 20 \
  --option 1 \
  --with_knn True
```

For MeshUp, the clustering step is optional if you pass feature files. Feature files are preferred because source and target are assigned in the same PartField feature space. Cluster label files are also supported, but object-local label IDs are not guaranteed to match across shapes.

Official PartField mesh inference commonly exports face-level arrays like:

```text
part_feat_blueberry_5k_upright_wheels_down_0_batch.npy
part_feat_bugatti-centodieci_5k_upright_wheels_down_0_batch.npy
```

Put them somewhere like:

```text
jobs_with_target_guidance/partfield_features/car_5k/
```

Keep the PartField input topology aligned with the MeshUp mesh you optimize. If PartField preprocessing/remeshing changes face or vertex counts, the feature array will no longer map directly to MeshUp vertices/faces.

The two dev jobs expect these three files:

```text
part_feat_blueberry_5k_upright_wheels_down_0_batch.npy
part_feat_bugatti-centodieci_5k_upright_wheels_down_0_batch.npy
part_feat_santa_fe_5k_upright_wheels_down_0_batch.npy
```

or the same names without `_batch`.

If you have an official PartField checkout and checkpoint, generate the three feature files with:

```bash
sbatch --export=ALL,PARTFIELD_REPO=/path/to/PartField,PARTFIELD_ENV=partfield \
  jobs_with_target_guidance/jobs/job_prepare_partfield_car_features.sh
```

The helper converts the three PLY meshes to OBJ for PartField mesh mode, runs `partfield_inference.py`, and copies `part_feat_*` outputs into `jobs_with_target_guidance/partfield_features/car_5k/`.

If you do not have the official checkout yet, fetch the repo and Objaverse checkpoint with:

```bash
jobs_with_target_guidance/scripts/setup_partfield_checkout.sh
```

This leaves the heavy model code in `external/PartField/` and prints the official conda environment command.

### Inspect PartField Parts

After feature extraction, make aligned labels and colored PLYs for blueberry, Santa Fe, and Bugatti:

```bash
sbatch jobs_with_target_guidance/jobs/job_dev_partfield_segment_car_features.sh
```

Outputs go to:

```text
jobs_with_target_guidance/partfield_segments/car_5k/
```

The important files are:

```text
colored/blueberry_partfield_12_parts.ply
colored/santa_fe_partfield_12_parts.ply
colored/bugatti_partfield_12_parts.ply
labels/blueberry_partfield_labels.npz
labels/santa_fe_partfield_labels.npz
labels/bugatti_partfield_labels.npz
summary.json
```

The labels are produced by `partfield_segment.py`, which clusters the PartField features jointly across all three cars. That means bucket `03`, for example, is fit as one shared feature-space cluster across the set rather than being an unrelated per-car id. PartField is class-agnostic, so the buckets do not come pre-named as `wheel` or `hood`; inspect the colored meshes and `summary.json` to decide which buckets correspond to wheels, front, rear, cabin, and so on.

If you pass these generated label files directly into MeshUp, also pass:

```bash
--partfield_labels_aligned
```

That tells the Chamfer loss to preserve the joint co-segmentation label ids instead of rematching target labels geometrically.

Then run:

```bash
sbatch jobs_with_target_guidance/jobs/job_dev_car_sources_to_bugatti_partfield_chamfer.sh
```

For the first pass, keep the jobs separate and only run blueberry as source:

```bash
sbatch jobs_with_target_guidance/jobs/job_dev_car_sources_to_bugatti_partfield_chamfer.sh
sbatch jobs_with_target_guidance/jobs/job_dev_blueberry_to_santafe_partfield_chamfer.sh
```

### PartField Knobs

- `target_mesh_partfield_chamfer_weight`: main strength of PartField bucket Chamfer.
- `partfield_source_features` / `partfield_target_features`: preferred PartField feature paths.
- `partfield_source_labels` / `partfield_target_labels`: optional clustered labels.
- `partfield_feature_mode`: `auto`, `face`, or `vertex`.
- `partfield_labels_aligned`: preserve label ids from joint co-segmentation outputs.
- `partfield_n_buckets`: source buckets to build from PartField features.
- `partfield_position_weight`: small normalized XYZ term to split repeated instances such as four wheels.
- `target_mesh_partfield_chamfer_points`: vertices sampled per active bucket.
- `target_mesh_chamfer_weight`: optional global Chamfer glue.

Debug labels are saved by default to:

```text
partfield_labels/bucket_labels.npz
partfield_labels/bucket_labels.json
```

TensorBoard logs:

```text
target_mesh_partfield_chamfer/
```
