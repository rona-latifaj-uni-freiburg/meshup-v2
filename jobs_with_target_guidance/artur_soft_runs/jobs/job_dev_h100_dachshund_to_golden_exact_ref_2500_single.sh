#!/bin/bash
#SBATCH --job-name=dev_d2gold_ref
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/artur_soft_runs/logs/dev_d2gold_ref_%j.out
#SBATCH --error=jobs_with_target_guidance/artur_soft_runs/logs/dev_d2gold_ref_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

VARIANT_ID=1
EPOCHS=${EPOCHS:-2500}
OUTPUT_ROOT=./jobs_with_target_guidance/artur_soft_runs/outputs_dev_dachshund_to_golden_exact_ref_aligned_pf
RUN_TAG=artur_soft_dev_h100_single

export PAIR_SLUG=dachshund_to_golden_retriever
export SOURCE=./experiments/dog_morphs/outputs/hound_to_dachshund_exp/mesh_final/mesh.obj
export TARGET=./experiments/dog_morphs/outputs/hound_to_golden_retriever_no_dino_exp/mesh_final/mesh.obj
export SOURCE_NAME=hound_to_dachshund
export TARGET_NAME=hound_to_golden_retriever_no_dino
export PROMPT="a dog"

export SOURCE_PARTFIELD_FEATURES=./jobs_with_target_guidance/partfield_features/no_dino_animals/part_feat_hound_to_dachshund_0_batch.npy
export SOURCE_PARTFIELD_LABELS=./jobs_with_target_guidance/partfield_segments/no_dino_animals_12/labels/hound_to_dachshund_partfield_labels.npz
export TARGET_PARTFIELD_FEATURES=./jobs_with_target_guidance/partfield_features/no_dino_animals/part_feat_hound_to_golden_retriever_no_dino_0_batch.npy
export TARGET_PARTFIELD_LABELS=./jobs_with_target_guidance/partfield_segments/no_dino_animals_12/labels/hound_to_golden_retriever_no_dino_partfield_labels.npz

bash ./jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh \
  "${VARIANT_ID}" \
  "${EPOCHS}" \
  "${OUTPUT_ROOT}" \
  "${RUN_TAG}"
