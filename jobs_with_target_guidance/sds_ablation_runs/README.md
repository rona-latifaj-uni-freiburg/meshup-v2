# No-SDS Car Ablations

This folder runs geometry-only target deformation ablations on dev H100 nodes.

## Ablations

- `chamfer_only`: global target mesh Chamfer, no SDS/diffusion, no DINO target guidance, no PartField.
- `partfield_chamfer`: global target mesh Chamfer plus hard PartField bucket Chamfer, no SDS/diffusion and no DINO target guidance.
- `partfield_chamfer_target_dino`: global target mesh Chamfer plus hard PartField bucket Chamfer and target DINO/render guidance, no SDS/diffusion.

## Pair IDs

- `0`: blueberry -> g_class
- `1`: f1_car -> f1_verstappen
- `2`: blueberry -> bugatti-centodieci
- `3`: mini_cooper -> g_class
- `4`: blueberry -> santa_fe

## Single-Job Submission

```bash
sbatch jobs_with_target_guidance/sds_ablation_runs/jobs/job_dev_h100_no_sds_ablation_2500_single.sh chamfer_only 0
sbatch jobs_with_target_guidance/sds_ablation_runs/jobs/job_dev_h100_no_sds_ablation_2500_single.sh partfield_chamfer 0
```

The wrapper uses `dev_gpu_h100`, one GPU, `00:30:00`, and `EPOCHS=2500` by default.
