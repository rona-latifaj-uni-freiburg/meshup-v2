#!/bin/bash
#SBATCH --job-name=prep_pf_cross_animals
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_pf_cross_animals_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/prep_pf_cross_animals_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

MESHUP_ROOT=/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
cd "${MESHUP_ROOT}"

BASE_DIR=${BASE_DIR:-${MESHUP_ROOT}/jobs_with_target_guidance/cross_animal_spike_runs}
mkdir -p \
  "${BASE_DIR}/logs" \
  "${BASE_DIR}/partfield/features/no_dino_animals" \
  "${BASE_DIR}/partfield/segments/no_dino_animals_12"

export PARTFIELD_DATA_SUBDIR=${PARTFIELD_DATA_SUBDIR:-data/meshup_no_dino_animals_cross_spike}
export PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME:-partfield_features/meshup_no_dino_animals_cross_spike}
export MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR:-${BASE_DIR}/partfield/features/no_dino_animals}
export SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR:-${BASE_DIR}/partfield/segments/no_dino_animals_12}
export N_BUCKETS=${N_BUCKETS:-12}
export POSITION_WEIGHT=${POSITION_WEIGHT:-0.05}
export NORMAL_WEIGHT=${NORMAL_WEIGHT:-0.0}

echo "======================================================"
echo "Cross-animal PartField preparation"
echo "BASE_DIR=${BASE_DIR}"
echo "MESHUP_FEATURE_DIR=${MESHUP_FEATURE_DIR}"
echo "SEGMENT_OUTPUT_DIR=${SEGMENT_OUTPUT_DIR}"
echo "PARTFIELD_RESULT_NAME=${PARTFIELD_RESULT_NAME}"
echo "START_TIME=$(date)"
echo "======================================================"

bash jobs_with_target_guidance/artur_soft_runs/jobs/job_prepare_partfield_no_dino_animals.sh

echo "END_TIME=$(date)"
echo "Cross-animal PartField features: ${MESHUP_FEATURE_DIR}"
echo "Cross-animal PartField labels: ${SEGMENT_OUTPUT_DIR}"
