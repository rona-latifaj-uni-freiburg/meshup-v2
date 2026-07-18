# DenseCorr3D Source-Stable Rerun

Submitted: 2026-07-02

## Failure Diagnosis

The first bucket-balanced chain created three jobs:

- `5698613`: panther -> cheetah finished the deformation and wrote
  `mesh_final_position.ply`, but the job exited nonzero when the chained
  `sbatch` for the next task hit `QOSMaxSubmitJobPerUserLimit`.
- `5700256`: bear -> panther failed at epoch 1 with
  `Nan in the forward pass of the POISSON SOLVE`.
- `5700506`: cheetah -> panther failed at epoch 1 with the same Poisson NaN.

Mesh checks showed the average bucket-balancing remesher can produce
source-side topology that is bad for the Poisson solve. The cheetah source also
had a tiny detached component in the original 5k mesh, which leaves an extra
nullspace in the source Laplacian.

## Fix

Use source-stable variants:

- The source mesh is not remeshed or smoothed.
- The target mesh is adjusted to match the source bucket counts.
- The cheetah source keeps only its largest connected component
  (`4992` vertices, `9996` faces).
- The chain script retries next-task `sbatch` submissions so a completed
  deformation is not marked failed just because the submit limit is temporarily
  full.

Source Poisson smoke tests passed for:

- bear -> panther source
- cheetah -> panther source
- panther -> cheetah source

## Variants

| Task | Deformation | Variant | Source vertices |
| ---: | --- | --- | ---: |
| 0 | panther -> cheetah | `panther_cheetah_srcstable5002` | 5002 |
| 1 | bear -> panther | `bear_panther_srcstable5002` | 5002 |
| 2 | cheetah -> panther | `cheetah_panther_srcstable4992` | 4992 |
| 3 | bear -> elephant | `bear_elephant_srcstable5002` | 5002 |
| 4 | elephant -> moose | `elephant_moose_srcstable5000` | 5000 |
| 5 | elephant -> giraffe | `elephant_giraffe_srcstable5000` | 5000 |
| 6 | giraffe -> elephant | `giraffe_elephant_srcstable4996` | 4996 |

All source meshes are single-component and below 6k vertices.

## Slurm

Submit script:

`jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_srcstable_requested_single.sh`

Initial job:

`5708185`

Current submit-time state:

`PENDING (Resources)`

Output root:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_srcstable_requested_5k_best_asym035`

Chain report:

`jobs_with_target_guidance/cross_animal_spike_runs/reports/densecorr3d_srcstable_requested_chain.txt`
