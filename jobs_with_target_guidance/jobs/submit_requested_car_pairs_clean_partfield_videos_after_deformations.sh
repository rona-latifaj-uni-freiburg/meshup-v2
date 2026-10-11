#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

OUTPUT_ROOT=${OUTPUT_ROOT:-jobs_with_target_guidance/outputs/requested_car_pairs_clean_partfield_4000}
VIDEO_JOB_SCRIPT=${VIDEO_JOB_SCRIPT:-jobs_with_target_guidance/jobs/job_dev_h100_make_single_requested_car_partfield_video.sh}
WAIT_TMUX_SESSION=${WAIT_TMUX_SESSION:-car_pf4000_dev_runner}
VIDEO_TMUX_SESSION=${VIDEO_TMUX_SESSION:-car_pf4000_video_runner}
LOG_DIR=${LOG_DIR:-jobs_with_target_guidance/logs}
VIDEO_OUTPUT_ROOT=${VIDEO_OUTPUT_ROOT:-${OUTPUT_ROOT}/videos}
LEDGER=${LEDGER:-${LOG_DIR}/car_pf4000_video_sequential_ledger.tsv}
RUN_LIST=${RUN_LIST:-${LOG_DIR}/car_pf4000_video_run_dirs.txt}
POLL_SECONDS=${POLL_SECONDS:-60}

mkdir -p "${LOG_DIR}" "${VIDEO_OUTPUT_ROOT}"

echo "Video submitter started at $(date)"
echo "Waiting for deformation tmux session ${WAIT_TMUX_SESSION} to finish."
while tmux has-session -t "${WAIT_TMUX_SESSION}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

echo "Deformation tmux session is gone; discovering completed runs."
find "${OUTPUT_ROOT}" -mindepth 3 -maxdepth 3 -path "*/colored_meshes/mesh_final_position.ply" -print \
  | sed 's#/colored_meshes/mesh_final_position.ply$##' \
  | sort > "${RUN_LIST}"

run_count=$(wc -l < "${RUN_LIST}" | tr -d ' ')
echo "Discovered ${run_count} completed run dirs."
if [[ "${run_count}" -eq 0 ]]; then
  echo "No completed runs found under ${OUTPUT_ROOT}; no videos submitted."
  exit 1
fi

if [[ ! -f "${LEDGER}" ]]; then
  printf "run_dir\tjob_id\tsubmitted_at\tstate\n" > "${LEDGER}"
fi

while IFS= read -r run_dir; do
  [[ -z "${run_dir}" ]] && continue
  echo "======================================================"
  echo "Submitting video job for ${run_dir}"
  job_id=$(sbatch --parsable \
    --partition=dev_gpu_h100 \
    --time=00:30:00 \
    --export=ALL,VIDEO_OUTPUT_ROOT="${VIDEO_OUTPUT_ROOT}" \
    "${VIDEO_JOB_SCRIPT}" \
    "${run_dir}")
  printf "%s\t%s\t%s\tSUBMITTED\n" "${run_dir}" "${job_id}" "$(date -Iseconds)" >> "${LEDGER}"
  echo "Submitted ${job_id}; waiting for it to leave Slurm queue."

  while squeue -h -j "${job_id}" >/tmp/car_pfvideo_squeue_${job_id}.txt 2>/tmp/car_pfvideo_squeue_${job_id}.err; do
    if [[ ! -s /tmp/car_pfvideo_squeue_${job_id}.txt ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done

  state=$(sacct -j "${job_id}" --format=State -P -n 2>/dev/null | awk -F '|' 'NF && $1 !~ /\.batch|\.extern/ { print $1; exit }')
  state=${state:-UNKNOWN}
  printf "%s\t%s\t%s\t%s\n" "${run_dir}" "${job_id}" "$(date -Iseconds)" "${state}" >> "${LEDGER}"
  echo "Video job ${job_id} finished with state ${state}"
done < "${RUN_LIST}"

echo "Video submitter finished at $(date)"
