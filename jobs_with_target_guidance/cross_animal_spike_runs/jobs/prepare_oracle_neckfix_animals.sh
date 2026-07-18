#!/bin/bash
# One-time, per-animal (not per-pair) data prep for the "oracle_neckfix" pipeline.
#
# For each requested DenseCorr3D animal:
#   1. densecorr3d_prepare_mesh_variants.py (existing, unmodified) -- transfers
#      groups.txt labels onto the full color mesh and decimates to ~5k verts.
#   2. densecorr3d_protect_small_buckets.py (new) -- raises any bucket below
#      the floor by split-only local subdivision.
#   3. densecorr3d_split_neck_bucket.py (new) -- splits the torso bucket into
#      torso + neck (8 -> 9 buckets).
#   4. densecorr3d_protect_small_buckets.py again -- tops up the fresh
#      torso/neck halves if either fell under the floor.
#
# Runs once; every pairing in run_oracle_neckfix_pair.sh reuses this output.
# CPU-only (pymeshlab decimation), no GPU needed -- run directly, not via sbatch.

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
source ./activate_meshup_new.sh

DENSECORR3D_ROOT=${DENSECORR3D_ROOT:-external/DenseCorr3D}
OUTPUT_DIR=${OUTPUT_DIR:-jobs_with_target_guidance/densematcher_runs/prepared/oracle_neckfix_20260702}
TARGET_VERTICES=${TARGET_VERTICES:-5000}
MIN_BUCKET_VERTICES=${MIN_BUCKET_VERTICES:-300}
MAX_VERTICES=${MAX_VERTICES:-7000}

ANIMALS=(
  "elephant:2d6b3_toy_animals_009"
  "moose:1d6d1_toy_animals_015"
  "giraffe:34fb4_toy_animals_019"
  "panther:071b8_toy_animals_017"
  "bear:96615_toy_animals_018"
  "cheetah:bdfd0_toy_animals_016"
)

mkdir -p "${OUTPUT_DIR}/meshes" "${OUTPUT_DIR}/labels" "${OUTPUT_DIR}/colored" "${OUTPUT_DIR}/summaries"

for entry in "${ANIMALS[@]}"; do
  name="${entry%%:*}"
  object_id="${entry#*:}"
  echo "======================================================"
  echo "Preparing ${name} (${object_id})"
  echo "======================================================"

  python -m jobs_with_target_guidance.densecorr3d_prepare_mesh_variants \
    --densecorr3d-root "${DENSECORR3D_ROOT}" \
    --category animals \
    --objects "${object_id}" \
    --output-dir "${OUTPUT_DIR}/base" \
    --variants 5k \
    --target-vertices "${TARGET_VERTICES}"

  base_mesh="${OUTPUT_DIR}/base/meshes/${object_id}_densecorr3d_5k.obj"
  base_labels="${OUTPUT_DIR}/base/labels/${object_id}_densecorr3d_5k_labels.npz"

  protected_mesh="${OUTPUT_DIR}/meshes/${name}_oracle_neckfix.obj"
  protected_labels_tmp="${OUTPUT_DIR}/labels/${name}_oracle_neckfix_8buckets_labels.npz"

  python jobs_with_target_guidance/densecorr3d_protect_small_buckets.py \
    --mesh "${base_mesh}" \
    --labels "${base_labels}" \
    --output-mesh "${protected_mesh}" \
    --output-labels "${protected_labels_tmp}" \
    --output-summary "${OUTPUT_DIR}/summaries/${name}_protect_pass1.json" \
    --min-bucket-vertices "${MIN_BUCKET_VERTICES}" \
    --max-vertices "${MAX_VERTICES}"

  neck_labels_tmp="${OUTPUT_DIR}/labels/${name}_oracle_neckfix_9buckets_unprotected_labels.npz"
  python jobs_with_target_guidance/densecorr3d_split_neck_bucket.py \
    --mesh "${protected_mesh}" \
    --labels "${protected_labels_tmp}" \
    --output-labels "${neck_labels_tmp}" \
    --output-summary "${OUTPUT_DIR}/summaries/${name}_neck_split.json"

  final_labels="${OUTPUT_DIR}/labels/${name}_oracle_neckfix_labels.npz"
  final_colored="${OUTPUT_DIR}/colored/${name}_oracle_neckfix_groups.ply"
  # Re-run floor protection on the neck-split labels; the mesh geometry is
  # already final at this point (this call must not need to add vertices in
  # practice, but stays here as a safety net -- see verification step).
  python jobs_with_target_guidance/densecorr3d_protect_small_buckets.py \
    --mesh "${protected_mesh}" \
    --labels "${neck_labels_tmp}" \
    --output-mesh "${OUTPUT_DIR}/meshes/${name}_oracle_neckfix_final.obj" \
    --output-labels "${final_labels}" \
    --output-colored "${final_colored}" \
    --output-summary "${OUTPUT_DIR}/summaries/${name}_protect_pass2.json" \
    --min-bucket-vertices "${MIN_BUCKET_VERTICES}" \
    --max-vertices "${MAX_VERTICES}"

  echo "Done: ${name} -> ${final_labels}"
done

echo "All animals prepared under ${OUTPUT_DIR}"
