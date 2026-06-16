#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPT_JOB="${SCRIPT_DIR}/job_dev_h100_no_dino_animal_pf_chamfer_single.sh"
VIZ_JOB="${SCRIPT_DIR}/job_dev_h100_no_dino_animal_displacement_viz.sh"
STATE_DIR="${RUNS_DIR}/logs"
LEDGER="${STATE_DIR}/no_dino_animals_4000_submitted_jobs.tsv"
LOG_FILE="${STATE_DIR}/no_dino_animals_4000_autosubmit.log"

MAX_USER_JOBS="${MAX_USER_JOBS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EPOCHS="${EPOCHS:-4000}"
JNEIGHBOR_WEIGHT="${JNEIGHBOR_WEIGHT:-1800}"
RUN_TAG="${RUN_TAG:-dev_h100_no_dino_animals_single}"

mkdir -p "${STATE_DIR}"
touch "${LOG_FILE}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

count_squeue() {
  local output
  if ! output="$(squeue -h -u "${USER}" -o '%i' 2>&1)"; then
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

pair_slug() {
  case "$1" in
    0) echo "bulldog_to_horse" ;;
    1) echo "bulldog_to_cat" ;;
    2) echo "bulldog_to_golden_retriever" ;;
    3) echo "bulldog_to_bear" ;;
    4) echo "dachshund_to_golden_retriever" ;;
    *) return 1 ;;
  esac
}

variant_slug() {
  case "$1" in
    pf_chamfer) echo "hard_partfield_chamfer_no_regularizer" ;;
    pf_chamfer_jneighbor) echo "hard_partfield_chamfer_jneighbor${JNEIGHBOR_WEIGHT}" ;;
    *) return 1 ;;
  esac
}

output_root() {
  case "$1" in
    pf_chamfer)
      echo "./jobs_with_target_guidance/artur_soft_runs/outputs_dev_no_dino_animals_4000/no_regularizer"
      ;;
    pf_chamfer_jneighbor)
      echo "./jobs_with_target_guidance/artur_soft_runs/outputs_dev_no_dino_animals_4000/jneighbor_${JNEIGHBOR_WEIGHT}"
      ;;
    *)
      return 1
      ;;
  esac
}

prep_job_id() {
  awk -F '\t' '$1 == "partfield" { print $5; exit }' "${LEDGER}"
}

lookup_job_id() {
  local kind="$1"
  local pair_id="$2"
  local mode="$3"
  awk -F '\t' -v kind="${kind}" -v pair_id="${pair_id}" -v mode="${mode}" \
    '$1 == kind && $2 == pair_id && $4 == mode { print $5; exit }' "${LEDGER}"
}

lookup_run_dir() {
  local kind="$1"
  local pair_id="$2"
  local mode="$3"
  awk -F '\t' -v kind="${kind}" -v pair_id="${pair_id}" -v mode="${mode}" \
    '$1 == kind && $2 == pair_id && $4 == mode { print $7; exit }' "${LEDGER}"
}

append_ledger() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${LEDGER}"
}

wait_for_slot() {
  local next="$1"
  local total_jobs
  while true; do
    total_jobs="$(count_squeue)"
    if (( total_jobs < MAX_USER_JOBS )); then
      return 0
    fi
    log "Waiting: total user jobs=${total_jobs}/${MAX_USER_JOBS}; next=${next}"
    sleep "${POLL_SECONDS}"
  done
}

submit_opt() {
  local pair_id="$1"
  local mode="$2"
  local prep_id="$3"
  local pair variant root output rc job_id run_dir

  pair="$(pair_slug "${pair_id}")"
  variant="$(variant_slug "${mode}")"
  root="$(output_root "${mode}")"

  wait_for_slot "opt ${pair} ${mode}"
  output="$(sbatch \
    --dependency=afterok:"${prep_id}" \
    --export=ALL,EPOCHS="${EPOCHS}",JNEIGHBOR_WEIGHT="${JNEIGHBOR_WEIGHT}",RUN_TAG="${RUN_TAG}" \
    "${OPT_JOB}" "${pair_id}" "${mode}" 2>&1)"
  rc=$?
  log "sbatch opt ${pair} ${mode}: ${output}"
  if [[ ${rc} -ne 0 || "${output}" != Submitted\ batch\ job\ * ]]; then
    return 1
  fi

  job_id="${output##* }"
  run_dir="${root}/${pair}_${RUN_TAG}_${variant}_${EPOCHS}ep_${job_id}"
  append_ledger "opt" "${pair_id}" "${pair}" "${mode}" "${job_id}" "afterok:${prep_id}" "${run_dir}"
  return 0
}

submit_viz() {
  local pair_id="$1"
  local mode="$2"
  local opt_id="$3"
  local run_dir="$4"
  local pair output rc job_id

  pair="$(pair_slug "${pair_id}")"
  wait_for_slot "viz ${pair} ${mode}"
  output="$(sbatch --dependency=afterok:"${opt_id}" "${VIZ_JOB}" "${run_dir}" 2>&1)"
  rc=$?
  log "sbatch viz ${pair} ${mode}: ${output}"
  if [[ ${rc} -ne 0 || "${output}" != Submitted\ batch\ job\ * ]]; then
    return 1
  fi

  job_id="${output##* }"
  append_ledger "viz" "${pair_id}" "${pair}" "${mode}" "${job_id}" "afterok:${opt_id}" "${run_dir}/displacement_viz"
  return 0
}

if [[ ! -f "${LEDGER}" ]]; then
  log "Missing submission ledger: ${LEDGER}"
  exit 1
fi

if [[ ! -x "${OPT_JOB}" || ! -x "${VIZ_JOB}" ]]; then
  log "Missing executable opt/viz job scripts"
  exit 1
fi

PREP_ID="$(prep_job_id)"
if [[ -z "${PREP_ID}" ]]; then
  log "No partfield prep job id found in ${LEDGER}"
  exit 1
fi

log "Starting no-DINO animal autosubmitter; prep=${PREP_ID}, max_user_jobs=${MAX_USER_JOBS}, poll=${POLL_SECONDS}s"

while true; do
  submitted_any=0
  missing_any=0

  for mode in pf_chamfer pf_chamfer_jneighbor; do
    for pair_id in 0 1 2 3 4; do
      opt_id="$(lookup_job_id opt "${pair_id}" "${mode}")"
      if [[ -z "${opt_id}" ]]; then
        missing_any=1
        if submit_opt "${pair_id}" "${mode}" "${PREP_ID}"; then
          submitted_any=1
        else
          log "Submit failed for opt pair_id=${pair_id} mode=${mode}; retrying after ${POLL_SECONDS}s"
          sleep "${POLL_SECONDS}"
        fi
        continue
      fi

      if [[ -z "$(lookup_job_id viz "${pair_id}" "${mode}")" ]]; then
        missing_any=1
        run_dir="$(lookup_run_dir opt "${pair_id}" "${mode}")"
        if submit_viz "${pair_id}" "${mode}" "${opt_id}" "${run_dir}"; then
          submitted_any=1
        else
          log "Submit failed for viz pair_id=${pair_id} mode=${mode}; retrying after ${POLL_SECONDS}s"
          sleep "${POLL_SECONDS}"
        fi
      fi
    done
  done

  if (( missing_any == 0 )); then
    log "All no-DINO animal opt/viz jobs have been submitted."
    exit 0
  fi

  if (( submitted_any == 0 )); then
    sleep "${POLL_SECONDS}"
  fi
done
