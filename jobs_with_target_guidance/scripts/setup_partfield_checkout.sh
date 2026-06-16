#!/bin/bash
set -euo pipefail

MESHUP_ROOT=${MESHUP_ROOT:-/pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2}
PARTFIELD_REPO=${PARTFIELD_REPO:-${MESHUP_ROOT}/external/PartField}
PARTFIELD_CKPT=${PARTFIELD_CKPT:-${PARTFIELD_REPO}/model/model_objaverse.ckpt}
PARTFIELD_REPO_URL=${PARTFIELD_REPO_URL:-https://github.com/nv-tlabs/PartField.git}
PARTFIELD_CKPT_URL=${PARTFIELD_CKPT_URL:-https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt}

mkdir -p "$(dirname "${PARTFIELD_REPO}")"

if [[ ! -d "${PARTFIELD_REPO}/.git" ]]; then
  git clone "${PARTFIELD_REPO_URL}" "${PARTFIELD_REPO}"
else
  git -C "${PARTFIELD_REPO}" pull --ff-only
fi

mkdir -p "$(dirname "${PARTFIELD_CKPT}")"
if [[ ! -f "${PARTFIELD_CKPT}" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -O "${PARTFIELD_CKPT}" "${PARTFIELD_CKPT_URL}"
  else
    curl -L -o "${PARTFIELD_CKPT}" "${PARTFIELD_CKPT_URL}"
  fi
fi

cat <<EOF
PartField checkout is ready:
  PARTFIELD_REPO=${PARTFIELD_REPO}
  PARTFIELD_CKPT=${PARTFIELD_CKPT}

Create the PartField environment once:
  jobs_with_target_guidance/scripts/create_partfield_env.sh

Note: the exported ${PARTFIELD_REPO}/environment.yml may fail on this cluster
because it pins CUDA 12.8 packages while the README setup targets CUDA 12.4.

Then extract MeshUp car features with:
  sbatch --export=ALL,PARTFIELD_REPO=${PARTFIELD_REPO},PARTFIELD_ENV=partfield \\
    jobs_with_target_guidance/jobs/job_prepare_partfield_car_features.sh
EOF
