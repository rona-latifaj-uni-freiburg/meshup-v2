#!/bin/bash
#SBATCH --job-name=analyze_cross_animals
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/analyze_cross_animals_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/analyze_cross_animals_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
source ./activate_meshup_new.sh

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/topomatch_vcorr}
REPORT_DIR=${REPORT_DIR:-${BASE_DIR}/reports}
mkdir -p "${REPORT_DIR}/per_run"

RUN_LIST="${REPORT_DIR}/completed_run_dirs.txt"
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort > "${RUN_LIST}"

while IFS= read -r run_dir; do
  [[ -d "${run_dir}/mesh_final" ]] || continue
  run_name=$(basename "${run_dir}")
  python jobs_with_target_guidance/analyze_displacement_outliers.py \
    "${run_dir}" \
    --top-k 12 \
    --output-name outlier_analysis
  python jobs_with_target_guidance/evaluate_target_pipeline.py \
    --output-dir "${run_dir}" \
    --samples 3000 \
    --part-samples 750
  {
    echo "# ${run_name}"
    echo
    echo "- Final mesh: \`${run_dir}/mesh_final/mesh.obj\`"
    echo "- Outlier report: \`${run_dir}/displacement_viz/outlier_analysis/summary.md\`"
    echo "- Evaluation JSON: \`${run_dir}/evaluation/target_metrics.json\`"
    echo "- Part metrics CSV: \`${run_dir}/evaluation/partfield_part_metrics.csv\`"
  } > "${REPORT_DIR}/per_run/${run_name}.md"
done < "${RUN_LIST}"

{
  echo "# Cross-animal spike sweep report"
  echo
  echo "Generated: $(date)"
  echo
  echo "Run directories:"
  sed 's/^/- `/' "${RUN_LIST}" | sed 's/$/`/'
  echo
  echo "Each run has:"
  echo "- \`mesh_final/mesh.obj\`: final deformation mesh"
  echo "- \`displacement_viz/outlier_analysis/summary.md\`: simple spike/outlier report"
  echo "- \`evaluation/target_metrics.json\`: global and PartField evaluation metrics"
  echo "- \`evaluation/partfield_part_metrics.csv\`: per-part metrics"
} > "${REPORT_DIR}/summary.md"

echo "Analysis complete: ${REPORT_DIR}/summary.md"
