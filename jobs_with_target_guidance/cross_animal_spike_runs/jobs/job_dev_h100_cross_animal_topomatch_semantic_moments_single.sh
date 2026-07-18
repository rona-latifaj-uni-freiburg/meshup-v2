#!/bin/bash
#SBATCH --job-name=dev_topo_semreg
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_topo_semreg_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/dev_topo_semreg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch $0 PAIR_ID"
  echo "PAIR_ID: 4 bulldog->cat, 5 cat->bulldog, 6 bulldog->dachshund, 7 dachshund->bulldog"
  exit 2
fi

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

PAIR_ID="$1"
BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
EPOCHS=${EPOCHS:-2500}
RUN_TAG=${RUN_TAG:-topomatch_semantic_moments_dev_h100}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/topomatch_semantic_moments}

mkdir -p "${OUTPUT_ROOT}" "${BASE_DIR}/logs"

# Keep the stable same-topology deformation as the primary body constraint.
export TARGET_VERTEX_CORRESPONDENCE_WEIGHT=${TARGET_VERTEX_CORRESPONDENCE_WEIGHT:-20000.0}
export TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=${TARGET_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS:-0}

# Do not use the failed hard semantic vertex map. It is too local and creates spikes.
export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT=${TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WEIGHT:-0.0}
export TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS=${TARGET_SEMANTIC_VERTEX_CORRESPONDENCE_WARMUP_EPOCHS:-0}

# No global nearest-neighbor Chamfer; it has no part awareness and can pull isolated vertices.
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=${GLOBAL_CHAMFER_WEIGHT_OVERRIDE:-0.0}

# Semantic guidance is low-frequency only: per-PartField bucket centroid, spread,
# and coordinate-profile matching. Hard/soft bucket Chamfer and fixed anchors are disabled.
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-1000.0}
export PARTFIELD_LABELS_ALIGNED=${PARTFIELD_LABELS_ALIGNED:-1}
export PARTFIELD_MIN_POINTS=${PARTFIELD_MIN_POINTS:-48}
export PARTFIELD_GUIDANCE_MODE=${PARTFIELD_GUIDANCE_MODE:-hard}
export PARTFIELD_HARD_WEIGHT=${PARTFIELD_HARD_WEIGHT:-0.0}
export PARTFIELD_SOFT_WEIGHT=${PARTFIELD_SOFT_WEIGHT:-0.0}
export PARTFIELD_MOMENT_WEIGHT=${PARTFIELD_MOMENT_WEIGHT:-1.0}
export PARTFIELD_MOMENT_EXTENT_WEIGHT=${PARTFIELD_MOMENT_EXTENT_WEIGHT:-0.45}
export PARTFIELD_PROFILE_WEIGHT=${PARTFIELD_PROFILE_WEIGHT:-0.25}
export PARTFIELD_PROFILE_BINS=${PARTFIELD_PROFILE_BINS:-9}
export PARTFIELD_PROFILE_TRIM=${PARTFIELD_PROFILE_TRIM:-0.10}
export PARTFIELD_ANCHOR_WEIGHT=${PARTFIELD_ANCHOR_WEIGHT:-0.0}
export PARTFIELD_ANCHOR_SEMANTIC_WEIGHT=${PARTFIELD_ANCHOR_SEMANTIC_WEIGHT:-0.0}
export PARTFIELD_CONTAINMENT_WEIGHT=${PARTFIELD_CONTAINMENT_WEIGHT:-0.0}

# Mild local regularizers remain as spike guards, not as the main deformation prior.
export DEFORMATION_PARAMETERIZATION=${DEFORMATION_PARAMETERIZATION:-vertex}
export EDGE_STRETCH_WEIGHT=${EDGE_STRETCH_WEIGHT:-100.0}
export EDGE_STRETCH_THRESHOLD=${EDGE_STRETCH_THRESHOLD:-1.35}
export EDGE_STRETCH_MAX_WEIGHT=${EDGE_STRETCH_MAX_WEIGHT:-1.0}
export EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-250.0}
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.25}
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=${EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT:-1.0}

export ENABLE_SOURCE_DINO_LOSS=${ENABLE_SOURCE_DINO_LOSS:-0}
export ENABLE_TARGET_DINO_GUIDANCE=${ENABLE_TARGET_DINO_GUIDANCE:-0}
export RUN_EVAL=${RUN_EVAL:-1}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-250}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-250}
export EXTRA_LOG_EPOCHS=${EXTRA_LOG_EPOCHS:-"1 10 25 50 100 250 500 1000 1500 2000 2500"}
export VARIANT_SUFFIX=${VARIANT_SUFFIX:-topomatch20000_semantic_moments_pf1000_min48_estretch100_jump250}

bash "${BASE_DIR}/jobs/run_cross_animal_best_pair.sh" \
  "${PAIR_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
