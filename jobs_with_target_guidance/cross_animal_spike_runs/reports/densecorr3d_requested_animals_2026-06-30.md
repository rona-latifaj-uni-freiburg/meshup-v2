# DenseCorr3D Requested Animals Run

Submitted: 2026-06-30

## Request

Run hard partfield-chamfer deformations, using the recipe that produced:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/cat_to_dachshund_catdach_only_pf08_4000_hard_partfield_chamfer_only_catdach_only_08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5168193`

Pairs:

1. giraffe -> elephant
2. elephant -> giraffe
3. bear -> cheetah
4. cheetah -> bear
5. panther -> bear
6. bear -> panther
7. moose -> elephant
8. elephant -> moose

Each pair is run on both `full` DenseCorr3D color meshes and `5k` topology-preserving decimations.

## Animal IDs

| Animal | DenseCorr3D object |
| --- | --- |
| elephant | `2d6b3_toy_animals_009` |
| moose | `1d6d1_toy_animals_015` |
| giraffe | `34fb4_toy_animals_019` |
| panther | `071b8_toy_animals_017` |
| bear | `96615_toy_animals_018` |
| cheetah | `bdfd0_toy_animals_016` |

## Prepared Assets

Prepared directory:

`jobs_with_target_guidance/densematcher_runs/prepared/requested_animals_20260630`

The preparation uses DenseCorr3D `groups.txt` as aligned semantic labels, transfers labels to each full `color_mesh.obj`, and creates `full` plus `5k` mesh/label variants.

Validation summary:

| Object | Full vertices | 5k vertices | Groups | Min 5k group vertices |
| --- | ---: | ---: | ---: | ---: |
| `2d6b3_toy_animals_009` | 190950 | 5000 | 8 | 314 |
| `1d6d1_toy_animals_015` | 179841 | 5002 | 8 | 174 |
| `34fb4_toy_animals_019` | 130100 | 4996 | 8 | 87 |
| `071b8_toy_animals_017` | 145127 | 5002 | 8 | 73 |
| `96615_toy_animals_018` | 253787 | 5002 | 8 | 99 |
| `bdfd0_toy_animals_016` | 217836 | 4996 | 8 | 61 |

All prepared label arrays match their mesh vertex counts.

## Pipeline Notes

The older prepared folder named `panther_bear` actually contains `071b8_toy_animals_017` and `bdfd0_toy_animals_016`. For this request, `bdfd0` is treated as cheetah and `96615` as bear, so the new prepared directory avoids reusing that misleading folder name.

The run script uses label-only DenseCorr3D semantics:

- `PARTFIELD_USE_FEATURES=0`
- `PARTFIELD_N_BUCKETS=8`
- `PARTFIELD_LABELS_ALIGNED=1`

The hard chamfer recipe mirrors the referenced output:

- `PARTFIELD_CHAMFER_WEIGHT=8000`
- `PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.35`
- `PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0`
- `DEFORMATION_PARAMETERIZATION=jacobian`
- `JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000`
- `EDGE_DISPLACEMENT_JUMP_WEIGHT=500`
- `EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.2`
- `EPOCHS=4000`

## Slurm

Array submission was rejected by `QOSMaxSubmitJobPerUserLimit`, so the jobs are launched as a serial chain using:

`jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_requested_animals_single.sh`

Initial chained job:

`5546194`

Each chained job requests:

- partition: `dev_gpu_h100`
- GPU: `1`
- time: `00:30:00`
- CPUs: `8`
- memory: `50G`

Chain order:

| Task | Pair | Variant |
| ---: | --- | --- |
| 0 | giraffe -> elephant | full |
| 1 | giraffe -> elephant | 5k |
| 2 | elephant -> giraffe | full |
| 3 | elephant -> giraffe | 5k |
| 4 | bear -> cheetah | full |
| 5 | bear -> cheetah | 5k |
| 6 | cheetah -> bear | full |
| 7 | cheetah -> bear | 5k |
| 8 | panther -> bear | full |
| 9 | panther -> bear | 5k |
| 10 | bear -> panther | full |
| 11 | bear -> panther | 5k |
| 12 | moose -> elephant | full |
| 13 | moose -> elephant | 5k |
| 14 | elephant -> moose | full |
| 15 | elephant -> moose | 5k |

Chain progress is appended to:

`jobs_with_target_guidance/cross_animal_spike_runs/reports/densecorr3d_requested_animals_chain.txt`

Outputs go under:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_requested_animals_best_asym035`
