#!/usr/bin/env bash
# Batch process an image folder: validate, mask, mesh (5k), component-filter, queue

set -euo pipefail

BASE_PATH="${BASE_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DIR="${SOURCE_DIR:-${BASE_PATH}/new_images}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_IMAGES="${MAX_IMAGES:-0}"
MASK_TIME="${MASK_TIME:-00:20:00}"
MESH_TIME="${MESH_TIME:-00:30:00}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
MASK_SCRIPT="${BASE_PATH}/scripts/slurm_sam2_mask_adaptive.sh"
MESH_SCRIPT="${BASE_PATH}/scripts/slurm_sam3d_reconstruct_5k_adaptive.sh"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[ERROR] source image folder not found: $SOURCE_DIR"
  exit 1
fi

mkdir -p "${BASE_PATH}/logs"

echo "======================================"
echo "New Images Batch Processing"
echo "======================================"
echo "[INFO] Source: $SOURCE_DIR"
echo "[INFO] Skip existing outputs: $SKIP_EXISTING"
echo "[INFO] Dry run: $DRY_RUN"
echo "[INFO] Max images: $MAX_IMAGES"
echo "[INFO] Partition: ${PARTITION:-<default>}"
echo "[INFO] Account: ${ACCOUNT:-<default>}"
echo ""

declare -a SBATCH_TARGET_ARGS=()
if [[ -n "$PARTITION" ]]; then
  SBATCH_TARGET_ARGS+=(--partition="$PARTITION")
fi
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_TARGET_ARGS+=(--account="$ACCOUNT")
fi

# Collect valid images
declare -a images
declare -a job_ids_mask
declare -a job_ids_mesh

echo "[STEP 1] Validating and collecting images..."
SOURCE_DIR="$SOURCE_DIR" SKIP_EXISTING="$SKIP_EXISTING" BASE_PATH="$BASE_PATH" MAX_IMAGES="$MAX_IMAGES" python3 << 'PYEOF'
import os
from pathlib import Path
from PIL import Image

source_dir = Path(os.environ['SOURCE_DIR'])
base_path = Path(os.environ['BASE_PATH'])
skip_existing = os.environ.get('SKIP_EXISTING', '1') == '1'
max_images = int(os.environ.get('MAX_IMAGES', '0'))
valid = []

for img_file in sorted(source_dir.glob('*')):
    if not img_file.is_file():
        continue

    mesh_file = base_path / 'sam3D' / 'mesh' / f'{img_file.stem}.ply'
    if skip_existing and mesh_file.exists():
        print(f"  - {img_file.name} - SKIPPED (mesh exists)")
        continue

    try:
        img = Image.open(img_file)
        img.load()
        print(f"  ✓ {img_file.name}")
        valid.append(img_file.name)
        if max_images and len(valid) >= max_images:
            break
    except Exception as e:
        print(f"  ✗ {img_file.name} - SKIPPED ({type(e).__name__})")

print(f"\nValid images: {len(valid)}")
if valid:
    for name in valid:
        print(f"VALID:{name}")
PYEOF

# Arrays for tracking job IDs
declare -a mask_job_ids
declare -a mesh_job_ids

echo ""
echo "[STEP 2] Queuing mask generation jobs..."
cd "$BASE_PATH"
for img_name in $(SOURCE_DIR="$SOURCE_DIR" SKIP_EXISTING="$SKIP_EXISTING" BASE_PATH="$BASE_PATH" MAX_IMAGES="$MAX_IMAGES" python3 << 'PYEOF'
import os
from pathlib import Path
from PIL import Image

source_dir = Path(os.environ['SOURCE_DIR'])
base_path = Path(os.environ['BASE_PATH'])
skip_existing = os.environ.get('SKIP_EXISTING', '1') == '1'
max_images = int(os.environ.get('MAX_IMAGES', '0'))
valid_count = 0
for img_file in sorted(source_dir.glob('*')):
    if not img_file.is_file():
        continue
    if skip_existing and (base_path / 'sam3D' / 'mesh' / f'{img_file.stem}.ply').exists():
        continue
    try:
        img = Image.open(img_file)
        img.load()
        print(img_file.name)
        valid_count += 1
        if max_images and valid_count >= max_images:
            break
    except:
        pass
PYEOF
); do
  cp -f "${SOURCE_DIR}/${img_name}" "${BASE_PATH}/image/${img_name}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${BASE_PATH}/mask/${img_name}" ]]; then
    echo "  Reusing existing mask for: $img_name"
    mask_job_ids+=("")
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY RUN] Would queue mask for: $img_name"
    mask_job_ids+=("DRYRUN")
  else
    echo "  Queueing mask for: $img_name"
    mask_job_output=$(sbatch \
      --parsable \
      "${SBATCH_TARGET_ARGS[@]}" \
      --time="${MASK_TIME}" \
      --export="ALL,BASE_PATH=${BASE_PATH},IMAGE_NAME=${img_name}" \
      "${MASK_SCRIPT}" 2>&1)
    mask_job_id=$(echo "$mask_job_output" | grep -oE '[0-9]+' | head -1)
    mask_job_ids+=("$mask_job_id")
    echo "    Job ID: $mask_job_id"
  fi
done

echo ""
echo "[STEP 3] Queuing mesh reconstruction jobs (with mask dependencies)..."
for i in "${!mask_job_ids[@]}"; do
  # Get image name in same order
  img_name=$(SOURCE_DIR="$SOURCE_DIR" SKIP_EXISTING="$SKIP_EXISTING" BASE_PATH="$BASE_PATH" MAX_IMAGES="$MAX_IMAGES" python3 << 'PYEOF' | sed -n "$((i+1))p"
import os
from pathlib import Path
from PIL import Image
source_dir = Path(os.environ['SOURCE_DIR'])
base_path = Path(os.environ['BASE_PATH'])
skip_existing = os.environ.get('SKIP_EXISTING', '1') == '1'
max_images = int(os.environ.get('MAX_IMAGES', '0'))
valid_count = 0
for img_file in sorted(source_dir.glob('*')):
    if not img_file.is_file():
        continue
    if skip_existing and (base_path / 'sam3D' / 'mesh' / f'{img_file.stem}.ply').exists():
        continue
    try:
        img = Image.open(img_file)
        img.load()
        print(img_file.name)
        valid_count += 1
        if max_images and valid_count >= max_images:
            break
    except:
        pass
PYEOF
)
  
  mask_dep="${mask_job_ids[$i]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY RUN] Would queue mesh for: $img_name"
    mesh_job_ids+=("DRYRUN")
    continue
  elif [[ -n "$mask_dep" ]]; then
    dependency_arg=(--dependency="afterok:${mask_dep}")
    echo "  Queueing mesh for: $img_name (depends on mask job $mask_dep)"
  else
    dependency_arg=()
    echo "  Queueing mesh for: $img_name (mask already exists)"
  fi

  mesh_job_output=$(sbatch \
    --parsable \
    "${SBATCH_TARGET_ARGS[@]}" \
    --time="${MESH_TIME}" \
    "${dependency_arg[@]}" \
    --export="ALL,BASE_PATH=${BASE_PATH},IMAGE_NAME=${img_name}" \
    "${MESH_SCRIPT}" 2>&1)
  mesh_job_id=$(echo "$mesh_job_output" | grep -oE '[0-9]+' | head -1)
  mesh_job_ids+=("$mesh_job_id")
  echo "    Job ID: $mesh_job_id"
done

echo ""
echo "======================================"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DONE] Dry-run preview complete."
else
  echo "[DONE] Batch queued successfully!"
fi
echo "======================================"
echo "Mask jobs: ${#mask_job_ids[@]}"
echo "Mesh jobs: ${#mesh_job_ids[@]}"
echo ""
echo "Monitor with: squeue --user=\$USER | grep sam3d"
echo "View logs with: tail -f logs/sam3d-*.out"
