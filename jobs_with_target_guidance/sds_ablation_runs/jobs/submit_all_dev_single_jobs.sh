#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

JOB=./jobs_with_target_guidance/sds_ablation_runs/jobs/job_dev_h100_no_sds_ablation_2500_single.sh

for ablation in chamfer_only partfield_chamfer; do
  for task_id in 0 1 2 3 4; do
    echo "Submitting ${ablation} task ${task_id}"
    sbatch "${JOB}" "${ablation}" "${task_id}"
  done
done
