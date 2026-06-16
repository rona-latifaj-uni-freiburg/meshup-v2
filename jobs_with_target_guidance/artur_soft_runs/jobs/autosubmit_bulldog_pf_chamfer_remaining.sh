#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SPEC_FILE="${SCRIPT_DIR}/bulldog_pf_chamfer_remaining_specs.tsv"
SINGLE_JOB="${SCRIPT_DIR}/job_dev_h100_bulldog_pf_chamfer_sweep_single.sh"
STATE_DIR="${RUNS_DIR}/logs"
LOG_FILE="${STATE_DIR}/bulldog_pf_chamfer_autosubmit.log"
LEDGER_KEYS="${STATE_DIR}/bulldog_pf_chamfer_submitted_keys.tsv"
LEDGER_JOBS="${STATE_DIR}/bulldog_pf_chamfer_submitted_jobs.tsv"

MAX_USER_JOBS="${MAX_USER_JOBS:-4}"
MAX_BDOG_JOBS="${MAX_BDOG_JOBS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${STATE_DIR}"
touch "${LEDGER_KEYS}" "${LEDGER_JOBS}" "${LOG_FILE}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

count_squeue() {
  local count_args=("$@")
  local output
  if ! output="$(squeue -h "${count_args[@]}" -o '%i' 2>&1)"; then
    log "squeue failed: ${output}"
    printf '999999\n'
    return
  fi
  if [[ -z "${output}" ]]; then
    printf '0\n'
    return
  fi
  printf '%s\n' "${output}" | wc -l | tr -d ' '
}

next_spec() {
  local mode variant weight key
  while read -r mode variant weight _; do
    [[ -z "${mode:-}" ]] && continue
    [[ "${mode}" == \#* ]] && continue
    key="${mode} ${variant} ${weight}"
    if ! grep -Fqx "${key}" "${LEDGER_KEYS}"; then
      printf '%s\t%s\t%s\n' "${mode}" "${variant}" "${weight}"
      return 0
    fi
  done < "${SPEC_FILE}"
  return 1
}

submit_spec() {
  local mode="$1"
  local variant="$2"
  local weight="$3"
  local key="${mode} ${variant} ${weight}"
  local output rc job_id

  output="$(sbatch "${SINGLE_JOB}" "${mode}" "${variant}" "${weight}" 2>&1)"
  rc=$?
  log "sbatch ${mode} ${variant} ${weight}: ${output}"

  if [[ ${rc} -eq 0 && "${output}" == Submitted\ batch\ job\ * ]]; then
    job_id="${output##* }"
    printf '%s\n' "${key}" >> "${LEDGER_KEYS}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${job_id}" "${mode}" "${variant}" "${weight}" >> "${LEDGER_JOBS}"
    return 0
  fi

  return 1
}

if [[ ! -f "${SPEC_FILE}" ]]; then
  log "Missing spec file: ${SPEC_FILE}"
  exit 1
fi

if [[ ! -x "${SINGLE_JOB}" ]]; then
  log "Missing executable job script: ${SINGLE_JOB}"
  exit 1
fi

log "Starting Bulldog PartField/Chamfer autosubmitter; max_user_jobs=${MAX_USER_JOBS}, max_bdog_jobs=${MAX_BDOG_JOBS}, poll=${POLL_SECONDS}s"

while true; do
  if ! spec="$(next_spec)"; then
    log "All remaining Bulldog PartField/Chamfer specs have been submitted."
    exit 0
  fi

  IFS=$'\t' read -r mode variant weight <<< "${spec}"
  total_jobs="$(count_squeue -u "${USER}")"
  bdog_jobs="$(count_squeue -u "${USER}" -n dev_bdog_pfch)"

  if (( total_jobs >= MAX_USER_JOBS )); then
    log "Waiting: total user jobs=${total_jobs}/${MAX_USER_JOBS}; next=${mode} ${variant} ${weight}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if (( bdog_jobs >= MAX_BDOG_JOBS )); then
    log "Waiting: Bulldog jobs=${bdog_jobs}/${MAX_BDOG_JOBS}; next=${mode} ${variant} ${weight}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  if ! submit_spec "${mode}" "${variant}" "${weight}"; then
    log "Submit failed; will retry ${mode} ${variant} ${weight} after ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi

  sleep 5
done
