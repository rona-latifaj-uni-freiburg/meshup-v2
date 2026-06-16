#!/bin/bash
#SBATCH --job-name=dev_no_sds_ab
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/sds_ablation_runs/logs/dev_no_sds_ab_%j.out
#SBATCH --error=jobs_with_target_guidance/sds_ablation_runs/logs/dev_no_sds_ab_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch $0 ABLATION TASK_ID"
  echo "  ABLATION: chamfer_only | partfield_chamfer | partfield_chamfer_target_dino"
  echo "  TASK_ID: 0=blueberry_to_g_class, 1=f1_car_to_f1_verstappen, 2=blueberry_to_bugatti, 3=mini_cooper_to_g_class, 4=blueberry_to_santa_fe"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

ABLATION="$1"
TASK_ID="$2"
EPOCHS=${EPOCHS:-2500}
OUTPUT_ROOT=${OUTPUT_ROOT:-./jobs_with_target_guidance/sds_ablation_runs/outputs_dev}
RUN_TAG=${RUN_TAG:-dev_h100_single}

bash ./jobs_with_target_guidance/sds_ablation_runs/jobs/run_no_sds_ablation_pair.sh \
  "${ABLATION}" \
  "${TASK_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
