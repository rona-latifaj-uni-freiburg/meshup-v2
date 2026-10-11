#!/bin/bash
#SBATCH --job-name=dc3d_all_bb
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_all_animals_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_all_animals_%A_%a.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
PREPARED_DIR=${PREPARED_DIR:-./jobs_with_target_guidance/densematcher_runs/prepared/all_animals_20260803}
EPOCHS=${EPOCHS:-4000}
RUN_TAG=${RUN_TAG:-densecorr3d_all_animals_bucketbalanced_5k}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_all_animals_bucketbalanced_5k_best_asym035}
ANIMAL_SPLIT=${ANIMAL_SPLIT:-all}
TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}

case "${ANIMAL_SPLIT}" in
  all)
    OBJECTS=(
      071b8_toy_animals_017
      13cf7_toy_animals_055
      188f4_toy_animals_022
      1d6d1_toy_animals_015
      2d6b3_toy_animals_009
      34fb4_toy_animals_019
      4e94a_toy_animals_020
      96615_toy_animals_018
      bdfd0_toy_animals_016
      d1889_dinosaur_002
    )
    ;;
  test)
    OBJECTS=(
      188f4_toy_animals_022
      bdfd0_toy_animals_016
      2d6b3_toy_animals_009
      d1889_dinosaur_002
      13cf7_toy_animals_055
    )
    ;;
  train)
    OBJECTS=(
      4e94a_toy_animals_020
      96615_toy_animals_018
      1d6d1_toy_animals_015
      071b8_toy_animals_017
      34fb4_toy_animals_019
    )
    ;;
  *)
    echo "Unknown ANIMAL_SPLIT=${ANIMAL_SPLIT}. Expected all, test, or train."
    exit 2
    ;;
esac

N_OBJECTS=${#OBJECTS[@]}
TASKS_PER_SOURCE=$((N_OBJECTS - 1))
TOTAL_TASKS=$((N_OBJECTS * TASKS_PER_SOURCE))
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "TASK_ID=${TASK_ID} is outside valid range 0..$((TOTAL_TASKS - 1)) for ANIMAL_SPLIT=${ANIMAL_SPLIT}."
  exit 2
fi

SOURCE_INDEX=$((TASK_ID / TASKS_PER_SOURCE))
TARGET_RANK=$((TASK_ID % TASKS_PER_SOURCE))
if (( TARGET_RANK >= SOURCE_INDEX )); then
  TARGET_INDEX=$((TARGET_RANK + 1))
else
  TARGET_INDEX=${TARGET_RANK}
fi

SOURCE_OBJECT=${OBJECTS[${SOURCE_INDEX}]}
TARGET_OBJECT=${OBJECTS[${TARGET_INDEX}]}
SOURCE_SHORT=${SOURCE_OBJECT%%_*}
TARGET_SHORT=${TARGET_OBJECT%%_*}
MESH_VARIANT="all_${ANIMAL_SPLIT}_${SOURCE_SHORT}_to_${TARGET_SHORT}_bucketavg"

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs" "${BASE_DIR}/reports"

if [[ ! -f "${PREPARED_DIR}/meshes/${SOURCE_OBJECT}_densecorr3d_5k.obj" || ! -f "${PREPARED_DIR}/meshes/${TARGET_OBJECT}_densecorr3d_5k.obj" ]]; then
  echo "Missing base 5k prepared mesh under ${PREPARED_DIR}."
  echo "Run jobs_with_target_guidance/densecorr3d_prepare_mesh_variants.py for the selected animal objects first."
  exit 1
fi

if [[ ! -f "${PREPARED_DIR}/meshes/${SOURCE_OBJECT}_densecorr3d_${MESH_VARIANT}.obj" || ! -f "${PREPARED_DIR}/meshes/${TARGET_OBJECT}_densecorr3d_${MESH_VARIANT}.obj" ]]; then
  source ./activate_meshup_new.sh
  python jobs_with_target_guidance/densecorr3d_balance_bucket_counts.py \
    --prepared-dir "${PREPARED_DIR}" \
    --source-object "${SOURCE_OBJECT}" \
    --target-object "${TARGET_OBJECT}" \
    --base-variant 5k \
    --output-variant "${MESH_VARIANT}" \
    --target-mode average \
    --max-vertices 6000 \
    --min-bucket-vertices 12 \
    --smooth-steps 1 \
    --smooth-alpha 0.08
fi

{
  echo "task_id=${TASK_ID}"
  echo "animal_split=${ANIMAL_SPLIT}"
  echo "source_object=${SOURCE_OBJECT}"
  echo "target_object=${TARGET_OBJECT}"
  echo "mesh_variant=${MESH_VARIANT}"
  echo "epochs=${EPOCHS}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "started_at=$(date)"
} >> "${BASE_DIR}/reports/densecorr3d_all_animals_bucketbalanced_${ANIMAL_SPLIT}.txt"

export BASE_DIR PREPARED_DIR EPOCHS RUN_TAG OUTPUT_ROOT

bash "${BASE_DIR}/jobs/run_densecorr3d_object_pair.sh" \
  "${SOURCE_OBJECT}" \
  "${TARGET_OBJECT}" \
  "${MESH_VARIANT}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
