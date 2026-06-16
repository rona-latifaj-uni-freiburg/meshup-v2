# Artur Soft Chamfer Runs

This folder keeps the supervisor/Artur soft-assignment ablation separate from the
earlier latent-only experiments.

The implemented soft target assignment is:

```text
d_ij = ||x_i - y_j||^2 / sigma_x^2 + lambda * feature_distance(f_i, g_j)
w_ij = softmax_j(-d_ij / tau)
L_XY = sum_i sum_j w_ij * ||x_i - y_j||^2
```

In the code this is `partfield_guidance_mode=soft` with
`partfield_soft_match_space=hybrid`.

Dev variants:

```text
0: global_chamfer_only
1: hard_partfield_chamfer_only
2: artur_soft_partfield_chamfer_only
```

All three dev jobs use bulldog to dachshund with the prompt `a dog`, disable SDS
and target-render/DINO guidance, and run for 2500 epochs on `dev_gpu_h100`.

Submit one by one:

```bash
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_soft_2500_single.sh 0
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_soft_2500_single.sh 1
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_soft_2500_single.sh 2
```

Car pair IDs for the repeated dev ablation:

```text
0: blueberry_to_santa_fe
1: f1_car_to_f1_verstappen
2: mini_cooper_to_g_class
```

Use the same variant IDs as above:

```bash
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_car_2500_single.sh 0 0
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_car_2500_single.sh 0 1
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_car_2500_single.sh 0 2
```

Animal 8k pair IDs:

```text
0: hound_to_hippo
1: hippo_to_hound
2: hippo_to_bulldog
3: hound_to_dachshund
```

Animal 8k variant IDs:

```text
0: hard_partfield_chamfer_reg1000
1: artur_soft_partfield_chamfer_reg1000
```

Submit one job at a time:

```bash
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_h100_artur_animal_8k_single.sh 0 0
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_h100_artur_animal_8k_single.sh 0 1
```

Dev-node version of the same animal 8k runs:

```bash
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_animal_8k_single.sh 0 0
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_artur_animal_8k_single.sh 0 1
```
