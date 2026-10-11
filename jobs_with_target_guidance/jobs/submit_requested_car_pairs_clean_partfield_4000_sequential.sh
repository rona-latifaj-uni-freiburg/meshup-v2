#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

JOB_SCRIPT=${JOB_SCRIPT:-jobs_with_target_guidance/jobs/job_dev_h100_requested_car_pairs_clean_partfield_4000_array.sh}
SPEC_FILE=${SPEC_FILE:-jobs_with_target_guidance/jobs/requested_car_pairs_clean_partfield_4000_specs.tsv}
LOG_DIR=${LOG_DIR:-jobs_with_target_guidance/logs}
LEDGER=${LEDGER:-${LOG_DIR}/car_pf4000_dev_sequential_ledger.tsv}
TASK_IDS=${TASK_IDS:-}
POLL_SECONDS=${POLL_SECONDS:-60}

mkdir -p "${LOG_DIR}"

if [[ -z "${TASK_IDS}" ]]; then
  TASK_IDS=$(awk -F '\t' '
    BEGIN { n = 0 }
    /^[[:space:]]*#/ || NF == 0 { next }
    { printf "%s ", n; n++ }
  ' "${SPEC_FILE}")
fi

if [[ ! -f "${LEDGER}" ]]; then
  printf "task_id\tpair_slug\tbucket\tjob_id\tsubmitted_at\tstate\n" > "${LEDGER}"
fi

echo "Sequential dev submitter started at $(date)"
echo "JOB_SCRIPT=${JOB_SCRIPT}"
echo "SPEC_FILE=${SPEC_FILE}"
echo "LEDGER=${LEDGER}"
echo "TASK_IDS=${TASK_IDS}"

for task_id in ${TASK_IDS}; do
  if ! line=$(awk -F '\t' -v task_id="${task_id}" '
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
    echo "No spec row found for task ${task_id}; stopping."
    exit 2
  fi

  pair_slug=$(awk -F '\t' '{ print $1 }' <<< "${line}")
  bucket=$(awk -F '\t' '{ print $6 }' <<< "${line}")

  echo "======================================================"
  echo "Submitting task ${task_id}: ${pair_slug} pf${bucket}"
  job_id=$(sbatch --parsable \
    --array="${task_id}" \
    --partition=dev_gpu_h100 \
    --time=00:30:00 \
    "${JOB_SCRIPT}")
  submitted_at=$(date -Iseconds)
  printf "%s\t%s\t%s\t%s\t%s\tSUBMITTED\n" "${task_id}" "${pair_slug}" "${bucket}" "${job_id}" "${submitted_at}" >> "${LEDGER}"
  echo "Submitted ${job_id}; waiting for it to leave Slurm queue."

  while squeue -h -j "${job_id}" >/tmp/car_pf4000_squeue_${job_id}.txt 2>/tmp/car_pf4000_squeue_${job_id}.err; do
    if [[ ! -s /tmp/car_pf4000_squeue_${job_id}.txt ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done

  state=$(sacct -j "${job_id}" --format=State -P -n 2>/dev/null | awk -F '|' 'NF && $1 !~ /\.batch|\.extern/ { print $1; exit }')
  state=${state:-UNKNOWN}
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${task_id}" "${pair_slug}" "${bucket}" "${job_id}" "$(date -Iseconds)" "${state}" >> "${LEDGER}"
  echo "Task ${task_id} job ${job_id} finished with state ${state}"
done

echo "Sequential dev submitter finished at $(date)"
