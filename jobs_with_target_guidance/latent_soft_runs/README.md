# PartField Latent-Soft Runs

Array tasks:

0. `bulldog_to_dachshund`, prompt `a dog`
1. `dachshund_to_bulldog`, prompt `a dog`
2. `f1_car_to_f1_verstappen`
3. `f1_verstappen_to_f1_car`
4. `mini_cooper_to_g_class`
5. `red_truck1_to_red_truck2`

Jobs:

```bash
sbatch jobs_with_target_guidance/latent_soft_runs/jobs/job_dev_h100_latent_soft_2500.sh
sbatch jobs_with_target_guidance/latent_soft_runs/jobs/job_h100_latent_soft_8000.sh
```

Non-array submissions:

```bash
for task_id in 0 1 2 3 4 5; do
  sbatch jobs_with_target_guidance/latent_soft_runs/jobs/job_dev_h100_latent_soft_2500_single.sh "${task_id}"
done

for task_id in 0 1 2 3 4 5; do
  sbatch jobs_with_target_guidance/latent_soft_runs/jobs/job_h100_latent_soft_8000_single.sh "${task_id}"
done
```

Outputs go to:

```text
jobs_with_target_guidance/latent_soft_runs/outputs_dev
jobs_with_target_guidance/latent_soft_runs/outputs_full
```

The PartField term is configured as:

```text
partfield_guidance_mode=soft
partfield_soft_match_space=latent
partfield_soft_semantic_weight=1.0
partfield_soft_temperature=0.05
```
