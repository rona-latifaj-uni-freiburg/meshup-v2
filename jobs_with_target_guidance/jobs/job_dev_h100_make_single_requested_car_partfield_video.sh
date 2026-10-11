#!/bin/bash
#SBATCH --job-name=car_pfvideo
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/logs/car_pfvideo_%j.out
#SBATCH --error=jobs_with_target_guidance/logs/car_pfvideo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 RUN_DIR"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
source ./activate_meshup_new.sh
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

RUN_DIR="$1"
VIDEO_OUTPUT_ROOT=${VIDEO_OUTPUT_ROOT:-jobs_with_target_guidance/outputs/requested_car_pairs_clean_partfield_4000/videos}
RUN_NAME=$(basename "${RUN_DIR}")
MESH_PATH="${RUN_DIR}/colored_meshes/mesh_final_position.ply"
OUT_DIR="${VIDEO_OUTPUT_ROOT}/${RUN_NAME}"
MANIFEST="${OUT_DIR}/manifest.txt"

if [[ ! -f "${RUN_DIR}/config.yml" ]]; then
  echo "Missing run config: ${RUN_DIR}/config.yml"
  exit 1
fi
if [[ ! -f "${MESH_PATH}" ]]; then
  echo "Missing final colored mesh: ${MESH_PATH}"
  exit 1
fi

mkdir -p "${OUT_DIR}"
printf "%s\n" "${MESH_PATH}" > "${MANIFEST}"

echo "Rendering PartField videos for ${RUN_DIR}"
echo "MESH_PATH=${MESH_PATH}"
echo "OUT_DIR=${OUT_DIR}"

python -m jobs_with_target_guidance.make_cross_animal_showoff_videos \
  --manifest "${MANIFEST}" \
  --per-run-output-dir "${OUT_DIR}" \
  --only-per-run \
  --summary-output "${OUT_DIR}/summary.json" \
  --width 1280 \
  --height 720 \
  --fps 24 \
  --bitrate 10M \
  --start-final-frames-per-run 72 \
  --deformation-frames-per-run 144 \
  --start-final-normalize independent \
  --deformation-normalize independent

echo "Finished videos for ${RUN_DIR}"
