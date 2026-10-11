#!/bin/bash
#SBATCH --job-name=dc3d_pf_curr
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_pf_curriculum_%A_%a.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_densecorr3d_pf_curriculum_%A_%a.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
SPEC_FILE=${SPEC_FILE:-${BASE_DIR}/jobs/densecorr3d_partfield_curriculum_try_specs.tsv}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_partfield_curriculum_try_4000}
REPORT_FILE=${REPORT_FILE:-${BASE_DIR}/reports/densecorr3d_partfield_curriculum_try_outputs.tsv}
RUN_TAG_PREFIX=${RUN_TAG_PREFIX:-densecorr3d_pf_curriculum}
TASK_ID=${1:-${SLURM_ARRAY_TASK_ID:-}}

STAGE1_EPOCHS=${STAGE1_EPOCHS:-1000}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-2500}
STAGE3_EPOCHS=${STAGE3_EPOCHS:-500}

if [[ -z "${TASK_ID}" ]]; then
  echo "Missing task id. Submit as an array job or pass a zero-based task id."
  exit 2
fi

if ! LINE=$(awk -F '\t' -v task_id="${TASK_ID}" '
  BEGIN { n = 0; found = 0 }
  /^[[:space:]]*#/ || NF == 0 { next }
  {
    if (n == task_id) {
      print
      found = 1
      exit
    }
    n++
  }
  END { if (!found) exit 1 }
' "${SPEC_FILE}"); then
  echo "No spec row found for task id ${TASK_ID} in ${SPEC_FILE}"
  exit 2
fi

IFS=$'\t' read -r \
  PAIR_SLUG \
  SOURCE \
  TARGET \
  SOURCE_NAME \
  TARGET_NAME \
  COARSE_BUCKET \
  FINE_BUCKET \
  SOURCE_COARSE_LABELS \
  TARGET_COARSE_LABELS \
  SOURCE_FINE_LABELS \
  TARGET_FINE_LABELS \
  SOURCE_PARTFIELD_FEATURES \
  TARGET_PARTFIELD_FEATURES \
  PROMPT \
  BASELINE_PF \
  BASELINE_AUC \
  BASELINE_ERR <<< "${LINE}"

if [[ -z "${PAIR_SLUG}" || -z "${SOURCE}" || -z "${TARGET}" || -z "${COARSE_BUCKET}" || -z "${FINE_BUCKET}" ]]; then
  echo "Malformed spec row for task id ${TASK_ID}: ${LINE}"
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs" "${BASE_DIR}/reports"

source ./activate_meshup_new.sh

ORIGINAL_SOURCE="${SOURCE}"
JOB_STEM=${SLURM_JOB_ID:-manual}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}"
    exit 1
  fi
}

for path in \
  "${SOURCE}" \
  "${TARGET}" \
  "${SOURCE_COARSE_LABELS}" \
  "${TARGET_COARSE_LABELS}" \
  "${SOURCE_FINE_LABELS}" \
  "${TARGET_FINE_LABELS}" \
  "${SOURCE_PARTFIELD_FEATURES}" \
  "${TARGET_PARTFIELD_FEATURES}"; do
  require_file "${path}"
done

patch_stage_config() {
  local stage_out="$1"
  local stage_name="$2"
  local stage_input_mesh="$3"
  local stage_labels="$4"
  python - "${stage_out}" "${ORIGINAL_SOURCE}" "${stage_input_mesh}" "${stage_name}" "${PAIR_SLUG}" "${stage_labels}" "${BASELINE_PF}" "${BASELINE_AUC}" "${BASELINE_ERR}" <<'PY'
from pathlib import Path
import sys
import yaml

run_dir = Path(sys.argv[1])
config_path = run_dir / "config.yml"
cfg = yaml.safe_load(config_path.read_text()) or {}
cfg["staged_original_source_mesh"] = sys.argv[2]
cfg["staged_stage_input_mesh"] = sys.argv[3]
cfg["staged_stage_name"] = sys.argv[4]
cfg["staged_pair_slug"] = sys.argv[5]
cfg["staged_label_scale"] = sys.argv[6]
cfg["staged_baseline_pf"] = sys.argv[7]
cfg["staged_baseline_auc"] = float(sys.argv[8])
cfg["staged_baseline_err"] = float(sys.argv[9])
config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

reset_common_env() {
  export SOURCE_NAME TARGET_NAME PAIR_SLUG PROMPT
  export TARGET="${TARGET}"
  export PARTFIELD_USE_FEATURES=1
  export PARTFIELD_LABELS_ALIGNED=1
  export DEFORMATION_PARAMETERIZATION=jacobian
  export JACOBIAN_REG_WEIGHT=0.0
  export JACOBIAN_OUTLIER_WEIGHT=0.0
  export JACOBIAN_OUTLIER_POWER=4.0
  export DEFORMATION_GRAD_CLIP_NORM=1.0
  export TARGET_VERTEX_CORRESPONDENCE_WEIGHT=0.0
  export TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=0
  export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=0.0
  export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=0
  export SEMANTIC_VERTEX_CORRESPONDENCE_REBUILD_CACHE=0
  export TARGET_CHAMFER_WARMUP_EPOCHS=0
  export PARTFIELD_CHAMFER_WARMUP_EPOCHS=0
  export ENABLE_SOURCE_DINO_LOSS=0
  export ENABLE_TARGET_DINO_GUIDANCE=0
  export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.45
  export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=0.70
  export PARTFIELD_TGT_TO_SRC_ROBUST_SCALE=0.05
  export PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=0.75
  export PARTFIELD_HARD_SEMANTIC_WEIGHT=0.25
  export PARTFIELD_HARD_GEOMETRY_SIGMA=0.75
  export PARTFIELD_SEMANTIC_CONFIDENCE_MIN_SIMILARITY=-1.0
  export PARTFIELD_SEMANTIC_CONFIDENCE_MARGIN=0.0
  export PARTFIELD_SEMANTIC_CONFIDENCE_FLOOR=1.0
  export PARTFIELD_SEMANTIC_CONFIDENCE_POWER=1.0
  export PARTFIELD_UNBALANCED_TRANSPORT_WEIGHT=0.0
  export PARTFIELD_UNBALANCED_TRANSPORT_RHO=0.30
  export ARTUR_SOFT_MATCH_SPACE=hybrid
  export ARTUR_SOFT_POINTS=1024
  export ARTUR_SOFT_GEOMETRY_SIGMA=0.5
  export ARTUR_SOFT_SEMANTIC_WEIGHT=1.0
  export ARTUR_SOFT_TEMPERATURE=0.1
  export PARTFIELD_BALANCED_SINKHORN_ITERS=50
  export PARTFIELD_MIN_POINTS=12
  export PARTFIELD_PROFILE_BINS=9
  export PARTFIELD_PROFILE_TRIM=0.08
  export PARTFIELD_ANCHOR_GEOMETRY_SIGMA=0.35
  export PARTFIELD_ANCHOR_SEMANTIC_WEIGHT=0.50
  export LOG_INTERVAL_IM=250
  export SAVE_RENDERS_INTERVAL=250
  export SAVE_PCA_VISUALIZATION=0
  export PCA_INTERVAL=250
  export PCA_N_VIEWS=8
  export PCA_RENDER_RES=518
  export PCA_DINO_MODEL=dinov2_vitl14
  export PCA_USE_EXTRA_LOG_EPOCHS=0
  export EXTRA_LOG_EPOCHS=
}

run_stage() {
  local stage_name="$1"
  local stage_input_mesh="$2"
  local stage_bucket="$3"
  local source_labels="$4"
  local target_labels="$5"
  local epochs="$6"
  local run_tag="$7"
  local variant_suffix="$8"

  export SOURCE="${stage_input_mesh}"
  export SOURCE_PARTFIELD_LABELS="${source_labels}"
  export TARGET_PARTFIELD_LABELS="${target_labels}"
  export SOURCE_PARTFIELD_FEATURES TARGET_PARTFIELD_FEATURES
  export PARTFIELD_N_BUCKETS="${stage_bucket}"
  export VARIANT_SUFFIX="${variant_suffix}"

  local variant="hard_partfield_chamfer_only_${VARIANT_SUFFIX}"
  local out_dir="${OUTPUT_ROOT}/${PAIR_SLUG}_${run_tag}_${variant}_${epochs}ep_${JOB_STEM}"

  echo "======================================================"
  echo "Curriculum ${stage_name}: ${PAIR_SLUG}"
  echo "input=${stage_input_mesh}"
  echo "target=${TARGET}"
  echo "bucket=${stage_bucket}"
  echo "epochs=${epochs}"
  echo "out=${out_dir}"
  echo "======================================================"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN would execute run_artur_chamfer_ablation.sh"
    echo "${out_dir}"
    return 0
  fi

  bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
    1 \
    "${epochs}" \
    "${OUTPUT_ROOT}" \
    "${run_tag}"

  require_file "${out_dir}/mesh_final/mesh.obj"
  patch_stage_config "${out_dir}" "${stage_name}" "${stage_input_mesh}" "PF${stage_bucket}"
  echo "${out_dir}"
}

reset_common_env
export PARTFIELD_GUIDANCE_MODE=hard
export PARTFIELD_HARD_WEIGHT=1.0
export PARTFIELD_SOFT_WEIGHT=0.0
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=150.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=3500.0
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=2000.0
export EDGE_STRETCH_WEIGHT=50.0
export EDGE_STRETCH_THRESHOLD=2.4
export EDGE_STRETCH_MAX_WEIGHT=2.0
export EDGE_DISPLACEMENT_JUMP_WEIGHT=1500.0
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=0.95
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.0
export PARTFIELD_CONTAINMENT_WEIGHT=0.05
export PARTFIELD_CONTAINMENT_MARGIN=0.02
export PARTFIELD_CONTAINMENT_MAX_WEIGHT=1.0
export PARTFIELD_MOMENT_WEIGHT=0.08
export PARTFIELD_MOMENT_EXTENT_WEIGHT=0.70
export PARTFIELD_PROFILE_WEIGHT=0.10
export PARTFIELD_ANCHOR_WEIGHT=0.0

STAGE1_TAG="${RUN_TAG_PREFIX}_${PAIR_SLUG}_stage1_pf${COARSE_BUCKET}"
STAGE1_SUFFIX="${PAIR_SLUG}_stage1_coarse_pf${COARSE_BUCKET}_g150_pf3500_j2000_jump1500_edge50"
STAGE1_OUT=$(run_stage "stage1_coarse_global" "${ORIGINAL_SOURCE}" "${COARSE_BUCKET}" "${SOURCE_COARSE_LABELS}" "${TARGET_COARSE_LABELS}" "${STAGE1_EPOCHS}" "${STAGE1_TAG}" "${STAGE1_SUFFIX}" | tail -n 1)

reset_common_env
export PARTFIELD_GUIDANCE_MODE=hybrid
export PARTFIELD_HARD_WEIGHT=0.90
export PARTFIELD_SOFT_WEIGHT=0.20
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=75.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=6000.0
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1500.0
export EDGE_STRETCH_WEIGHT=35.0
export EDGE_STRETCH_THRESHOLD=2.2
export EDGE_STRETCH_MAX_WEIGHT=2.0
export EDGE_DISPLACEMENT_JUMP_WEIGHT=1200.0
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=0.90
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.0
export PARTFIELD_CONTAINMENT_WEIGHT=0.15
export PARTFIELD_CONTAINMENT_MARGIN=0.01
export PARTFIELD_CONTAINMENT_MAX_WEIGHT=1.5
export PARTFIELD_MOMENT_WEIGHT=0.20
export PARTFIELD_MOMENT_EXTENT_WEIGHT=1.00
export PARTFIELD_PROFILE_WEIGHT=0.35
export PARTFIELD_ANCHOR_WEIGHT=0.25

STAGE2_TAG="${RUN_TAG_PREFIX}_${PAIR_SLUG}_stage2_pf${FINE_BUCKET}"
STAGE2_SUFFIX="${PAIR_SLUG}_stage2_fine_pf${FINE_BUCKET}_hybrid_g75_pf6000_h090_s020_m020_p035_a025_c015_j1500_jump1200"
STAGE2_OUT=$(run_stage "stage2_fine_semantic" "${STAGE1_OUT}/mesh_final/mesh.obj" "${FINE_BUCKET}" "${SOURCE_FINE_LABELS}" "${TARGET_FINE_LABELS}" "${STAGE2_EPOCHS}" "${STAGE2_TAG}" "${STAGE2_SUFFIX}" | tail -n 1)

reset_common_env
export PARTFIELD_GUIDANCE_MODE=hybrid
export PARTFIELD_HARD_WEIGHT=0.85
export PARTFIELD_SOFT_WEIGHT=0.15
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=40.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=5000.0
export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.40
export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=0.70
export PARTFIELD_TGT_TO_SRC_ROBUST_SCALE=0.04
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=2200.0
export EDGE_STRETCH_WEIGHT=75.0
export EDGE_STRETCH_THRESHOLD=2.0
export EDGE_STRETCH_MAX_WEIGHT=2.5
export EDGE_DISPLACEMENT_JUMP_WEIGHT=1800.0
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=0.85
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.5
export PARTFIELD_CONTAINMENT_WEIGHT=0.10
export PARTFIELD_CONTAINMENT_MARGIN=0.012
export PARTFIELD_CONTAINMENT_MAX_WEIGHT=1.5
export PARTFIELD_MOMENT_WEIGHT=0.10
export PARTFIELD_MOMENT_EXTENT_WEIGHT=1.00
export PARTFIELD_PROFILE_WEIGHT=0.20
export PARTFIELD_ANCHOR_WEIGHT=0.20

STAGE3_TAG="${RUN_TAG_PREFIX}_${PAIR_SLUG}_stage3_pf${FINE_BUCKET}"
STAGE3_SUFFIX="${PAIR_SLUG}_stage3_refine_pf${FINE_BUCKET}_hybrid_g40_pf5000_h085_s015_m010_p020_a020_c010_j2200_jump1800_edge75"
STAGE3_OUT=$(run_stage "stage3_refine_regularized" "${STAGE2_OUT}/mesh_final/mesh.obj" "${FINE_BUCKET}" "${SOURCE_FINE_LABELS}" "${TARGET_FINE_LABELS}" "${STAGE3_EPOCHS}" "${STAGE3_TAG}" "${STAGE3_SUFFIX}" | tail -n 1)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Curriculum final output: ${STAGE3_OUT}"
  exit 0
fi

{
  if [[ ! -s "${REPORT_FILE}" ]]; then
    printf "task_id\tjob_id\tpair\tbaseline_pf\tbaseline_auc\tbaseline_err\tstage1\tstage2\tstage3_final\tfinished_at\n"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${TASK_ID}" \
    "${JOB_STEM}" \
    "${PAIR_SLUG}" \
    "${BASELINE_PF}" \
    "${BASELINE_AUC}" \
    "${BASELINE_ERR}" \
    "${STAGE1_OUT}" \
    "${STAGE2_OUT}" \
    "${STAGE3_OUT}" \
    "$(date -Is)"
} >> "${REPORT_FILE}"

echo "Curriculum final output: ${STAGE3_OUT}"
