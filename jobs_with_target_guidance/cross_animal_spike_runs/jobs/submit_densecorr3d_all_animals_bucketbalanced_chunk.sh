#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

START_TASK=${1:-0}
CHUNK_SIZE=${CHUNK_SIZE:-3}
MAX_TASK=${MAX_TASK:-89}
CONCURRENCY=${CONCURRENCY:-1}
ANIMAL_SPLIT=${ANIMAL_SPLIT:-all}
PREPARED_DIR=${PREPARED_DIR:-jobs_with_target_guidance/densematcher_runs/prepared/all_animals_20260803}
EPOCHS=${EPOCHS:-4000}
OUTPUT_ROOT=${OUTPUT_ROOT:-jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_all_animals_bucketbalanced_5k_best_asym035}
RUN_TAG=${RUN_TAG:-densecorr3d_all_animals_bucketbalanced_5k}

if [[ "${ANIMAL_SPLIT}" == "test" || "${ANIMAL_SPLIT}" == "train" ]]; then
  MAX_TASK=${MAX_TASK_OVERRIDE:-19}
fi

if (( START_TASK > MAX_TASK )); then
  echo "START_TASK=${START_TASK} is already beyond MAX_TASK=${MAX_TASK}; nothing to submit."
  exit 0
fi

END_TASK=$((START_TASK + CHUNK_SIZE - 1))
if (( END_TASK > MAX_TASK )); then
  END_TASK=${MAX_TASK}
fi

job_id=$(sbatch --parsable \
  --array="${START_TASK}-${END_TASK}%${CONCURRENCY}" \
  --export=ALL,ANIMAL_SPLIT="${ANIMAL_SPLIT}",PREPARED_DIR="${PREPARED_DIR}",EPOCHS="${EPOCHS}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_TAG="${RUN_TAG}" \
  jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_densecorr3d_all_animals_bucketbalanced_array.sh)

echo "submitted_job=${job_id}"
echo "submitted_tasks=${START_TASK}-${END_TASK}"
if (( END_TASK < MAX_TASK )); then
  echo "next_command=CHUNK_SIZE=${CHUNK_SIZE} CONCURRENCY=${CONCURRENCY} ANIMAL_SPLIT=${ANIMAL_SPLIT} bash $0 $((END_TASK + 1))"
fi
