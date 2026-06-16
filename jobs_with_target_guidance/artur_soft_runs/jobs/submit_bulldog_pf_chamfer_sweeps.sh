#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

JOB=jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_bulldog_pf_chamfer_sweep_single.sh
WEIGHTS=(100 500 1000 1800)
VARIANTS=(1 2)

echo "Submitting Bulldog->Dachshund PartField+Chamfer+DINO runs"
for variant_id in "${VARIANTS[@]}"; do
  printf "dino variant %s: " "${variant_id}"
  sbatch "${JOB}" dino "${variant_id}"
done

echo "Submitting Bulldog->Dachshund PartField+Chamfer identity-Jacobian regularizer sweep"
for weight in "${WEIGHTS[@]}"; do
  for variant_id in "${VARIANTS[@]}"; do
    printf "reg%s variant %s: " "${weight}" "${variant_id}"
    sbatch "${JOB}" reg "${variant_id}" "${weight}"
  done
done

echo "Submitting Bulldog->Dachshund PartField+Chamfer neighboring-face Jacobian smoothness sweep"
for weight in "${WEIGHTS[@]}"; do
  for variant_id in "${VARIANTS[@]}"; do
    printf "jneighbor%s variant %s: " "${weight}" "${variant_id}"
    sbatch "${JOB}" jneighbor "${variant_id}" "${weight}"
  done
done
