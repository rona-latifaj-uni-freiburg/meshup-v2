#!/bin/bash
#SBATCH --job-name=egg_ctrl_g
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:20:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/eggplant_control_global_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/eggplant_control_global_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

OUTPUT_ROOT=${OUTPUT_ROOT:-./jobs_with_target_guidance/cross_animal_spike_runs/outputs/controls_20260821_gpu_smoke}
EPOCHS=${EPOCHS:-1000}
RUN_TAG=${RUN_TAG:-controls_20260821_eggplant}

export SOURCE=external/DenseCorr3D/eggplant/691d0e4cb6a1409ea72d960d0ab144ea/simple_mesh.obj
export TARGET=external/DenseCorr3D/eggplant/334a3ab0383f49d1bd34575a36ddf4d6/simple_mesh.obj
export SOURCE_NAME=eggplant_m000_691d0e4c
export TARGET_NAME=eggplant_m001_334a3ab0
export PAIR_SLUG=eggpS0T1
export PROMPT="an eggplant"

export SOURCE_PARTFIELD_LABELS=jobs_with_target_guidance/partfield_segments/densecorr3d_categories/densecorr3d_nonanimals_test_20260806/eggplant/buckets_01/labels/eggplant_m000_691d0e4c_partfield_labels.npz
export TARGET_PARTFIELD_LABELS=jobs_with_target_guidance/partfield_segments/densecorr3d_categories/densecorr3d_nonanimals_test_20260806/eggplant/buckets_01/labels/eggplant_m001_334a3ab0_partfield_labels.npz
export SOURCE_PARTFIELD_FEATURES=jobs_with_target_guidance/partfield_features/densecorr3d_categories/densecorr3d_nonanimals_test_20260806/eggplant/part_feat_eggplant_m000_691d0e4c_0_batch.npy
export TARGET_PARTFIELD_FEATURES=jobs_with_target_guidance/partfield_features/densecorr3d_categories/densecorr3d_nonanimals_test_20260806/eggplant/part_feat_eggplant_m001_334a3ab0_0_batch.npy

export PARTFIELD_N_BUCKETS=1
export PARTFIELD_USE_FEATURES=1
export PARTFIELD_LABELS_ALIGNED=1
export GLOBAL_CHAMFER_WEIGHT_OVERRIDE=750.0
export PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=0.0

export DEFORMATION_PARAMETERIZATION=jacobian
export JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1000.0
export EDGE_DISPLACEMENT_JUMP_WEIGHT=500.0
export EDGE_DISPLACEMENT_JUMP_THRESHOLD=1.2
export EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=2.0
export ENABLE_SOURCE_DINO_LOSS=0
export ENABLE_TARGET_DINO_GUIDANCE=0
export SAVE_PCA_VISUALIZATION=0
export LOG_INTERVAL_IM=250
export SAVE_RENDERS_INTERVAL=250
export EXTRA_LOG_EPOCHS=
export VARIANT_SUFFIX="globalonly_control_${EPOCHS}ep"

mkdir -p "${OUTPUT_ROOT}" jobs_with_target_guidance/cross_animal_spike_runs/logs

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  0 \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
