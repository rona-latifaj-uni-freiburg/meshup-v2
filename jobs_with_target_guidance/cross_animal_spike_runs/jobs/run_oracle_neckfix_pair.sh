#!/bin/bash
# Pair runner for the "oracle_neckfix" recipe: DenseCorr3D groups.txt labels
# (not PartField) with a dedicated neck bucket and small-bucket vertex-count
# protection (see prepare_oracle_neckfix_animals.sh), combined with the
# *unmodified*, already-proven hard-bucket asymmetric Chamfer recipe from
# best_asym035_jump500 (source->target 0.35, target->source 1.0, jacobian
# neighbor smoothing 1000, edge-displacement-jump 500). No moment / profile /
# anchor / containment / robust-Chamfer / unbalanced-OT weight -- those were
# tried twice elsewhere in this project and made results worse both times.
#
# Delegates to the existing, unmodified shared runner
# jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh,
# exactly like every other recipe in this repo (run_cross_animal_best_pair.sh,
# run_densecorr3d_requested_animal_pair.sh, ...).

set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: $0 PAIR_ID EPOCHS OUTPUT_ROOT RUN_TAG"
  echo "PAIR_ID: 0 elephant->giraffe, 1 giraffe->elephant"
  exit 2
fi

PAIR_ID="$1"
EPOCHS="$2"
OUTPUT_ROOT="$3"
RUN_TAG="$4"

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-./jobs_with_target_guidance/cross_animal_spike_runs}
PREPARED_DIR=${PREPARED_DIR:-./jobs_with_target_guidance/densematcher_runs/prepared/oracle_neckfix_20260702}

animal_mesh() { echo "${PREPARED_DIR}/meshes/${1}_oracle_neckfix_final.obj"; }
animal_labels() { echo "${PREPARED_DIR}/labels/${1}_oracle_neckfix_labels.npz"; }

case "${PAIR_ID}" in
  0)
    export PAIR_SLUG=elephant_to_giraffe
    export SOURCE_NAME=elephant_oracle_neckfix
    export TARGET_NAME=giraffe_oracle_neckfix
    export SOURCE=$(animal_mesh elephant)
    export TARGET=$(animal_mesh giraffe)
    export SOURCE_PARTFIELD_LABELS=$(animal_labels elephant)
    export TARGET_PARTFIELD_LABELS=$(animal_labels giraffe)
    export PROMPT="a giraffe"
    ;;
  1)
    export PAIR_SLUG=giraffe_to_elephant
    export SOURCE_NAME=giraffe_oracle_neckfix
    export TARGET_NAME=elephant_oracle_neckfix
    export SOURCE=$(animal_mesh giraffe)
    export TARGET=$(animal_mesh elephant)
    export SOURCE_PARTFIELD_LABELS=$(animal_labels giraffe)
    export TARGET_PARTFIELD_LABELS=$(animal_labels elephant)
    export PROMPT="an elephant"
    ;;
  *)
    echo "Unknown PAIR_ID=${PAIR_ID}. Expected 0 or 1."
    exit 2
    ;;
esac

for path in "${SOURCE}" "${TARGET}" "${SOURCE_PARTFIELD_LABELS}" "${TARGET_PARTFIELD_LABELS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing prepared input: ${path}. Run prepare_oracle_neckfix_animals.sh first."
    exit 1
  fi
done

export PARTFIELD_USE_FEATURES=0
export PARTFIELD_N_BUCKETS=9
export PARTFIELD_LABELS_ALIGNED=1
export PARTFIELD_MIN_POINTS=${PARTFIELD_MIN_POINTS:-12}
export PARTFIELD_GUIDANCE_MODE=hard
export PARTFIELD_HARD_WEIGHT=1.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=${PARTFIELD_CHAMFER_WEIGHT_OVERRIDE:-8000.0}
export PARTFIELD_SOURCE_TO_TARGET_WEIGHT=${PARTFIELD_SOURCE_TO_TARGET_WEIGHT:-0.35}
export PARTFIELD_TARGET_TO_SOURCE_WEIGHT=${PARTFIELD_TARGET_TO_SOURCE_WEIGHT:-1.0}
export DEFORMATION_PARAMETERIZATION=jacobian
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=${JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT:-1000.0}
export EDGE_DISPLACEMENT_JUMP_WEIGHT=${EDGE_DISPLACEMENT_JUMP_WEIGHT:-500.0}
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=${EDGE_DISPLACEMENT_JUMP_THRESHOLD:-1.2}
export LOG_INTERVAL_IM=${LOG_INTERVAL_IM:-500}
export SAVE_RENDERS_INTERVAL=${SAVE_RENDERS_INTERVAL:-500}
export VARIANT_SUFFIX=${VARIANT_SUFFIX:-oracle_neckfix_asym035_jneighbor1000_jump500}

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  1 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
