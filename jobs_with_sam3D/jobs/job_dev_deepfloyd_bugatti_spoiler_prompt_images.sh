#!/bin/bash
#SBATCH --job-name=dev_df_bug_spoil
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_sam3D/logs/dev_df_bug_spoil_%j.out
#SBATCH --error=jobs_with_sam3D/logs/dev_df_bug_spoil_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=latifajrona@gmail.com

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_sam3D/logs jobs_with_sam3D/prompt_images/bugatti_spoiler

source ./activate_meshup_new.sh

OUT=jobs_with_sam3D/prompt_images/bugatti_spoiler/deepfloyd_${SLURM_JOB_ID:-manual}

python experiments/simple_meshes/deepfloyd_prompt_check.py \
  --out_dir "${OUT}" \
  --model_size XL \
  --seed 42 \
  --num_inference_steps 80 \
  --run_stage2 \
  --stage2_model L \
  --num_inference_steps_stage2 40 \
  --guidance_scale_stage2 4.5 \
  --stage2_noise_level 100 \
  --guidance_scale 8.0 \
  --height 64 \
  --width 64 \
  --dtype float16 \
  --prompts \
    "a sports car" \
    "a sports car with a rear spoiler" \
    "a bugatti centodieci" \
    "a bugatti centodieci with a rear spoiler"

echo "DeepFloyd Bugatti spoiler prompt images complete."
echo "Outputs: ${OUT}"
