#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2
mkdir -p jobs_with_target_guidance/artur_soft_runs/logs

PREP_JOB=jobs_with_target_guidance/artur_soft_runs/jobs/job_prepare_partfield_no_dino_animals.sh
OPT_JOB=jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_no_dino_animal_pf_chamfer_single.sh
VIZ_JOB=jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_no_dino_animal_displacement_viz.sh
EPOCHS=${EPOCHS:-4000}
JNEIGHBOR_WEIGHT=${JNEIGHBOR_WEIGHT:-1800}
RUN_TAG=${RUN_TAG:-dev_h100_no_dino_animals_single}
LOG=jobs_with_target_guidance/artur_soft_runs/logs/no_dino_animals_4000_submitted_jobs.tsv

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

job_id_from_sbatch() {
  awk '{print $4}'
}

echo "Submitting no-DINO animal PartField prep"
prep_submit=$(sbatch "${PREP_JOB}")
prep_job_id=$(printf "%s\n" "${prep_submit}" | job_id_from_sbatch)
printf "%s\n" "${prep_submit}"

{
  printf "kind\tpair_id\tpair\tmode\tjob_id\tdependency\trun_dir\n"
  printf "partfield\t-\t-\t-\t%s\t-\t%s\n" "${prep_job_id}" "jobs_with_target_guidance/partfield_segments/no_dino_animals_12"
} > "${LOG}"

for mode in pf_chamfer pf_chamfer_jneighbor; do
  for pair_id in 0 1 2 3 4; do
    pair=$(pair_slug "${pair_id}")
    variant=$(variant_slug "${mode}")
    root=$(output_root "${mode}")
    opt_submit=$(sbatch \
      --dependency=afterok:"${prep_job_id}" \
      --export=ALL,EPOCHS="${EPOCHS}",JNEIGHBOR_WEIGHT="${JNEIGHBOR_WEIGHT}",RUN_TAG="${RUN_TAG}" \
      "${OPT_JOB}" "${pair_id}" "${mode}")
    opt_job_id=$(printf "%s\n" "${opt_submit}" | job_id_from_sbatch)
    run_dir="${root}/${pair}_${RUN_TAG}_${variant}_${EPOCHS}ep_${opt_job_id}"
    printf "%s %s %s\n" "${pair}" "${mode}" "${opt_submit}"
    printf "opt\t%s\t%s\t%s\t%s\tafterok:%s\t%s\n" \
      "${pair_id}" "${pair}" "${mode}" "${opt_job_id}" "${prep_job_id}" "${run_dir}" >> "${LOG}"

    viz_submit=$(sbatch --dependency=afterok:"${opt_job_id}" "${VIZ_JOB}" "${run_dir}")
    viz_job_id=$(printf "%s\n" "${viz_submit}" | job_id_from_sbatch)
    printf "%s %s viz %s\n" "${pair}" "${mode}" "${viz_submit}"
    printf "viz\t%s\t%s\t%s\t%s\tafterok:%s\t%s/displacement_viz\n" \
      "${pair_id}" "${pair}" "${mode}" "${viz_job_id}" "${opt_job_id}" "${run_dir}" >> "${LOG}"
  done
done

echo "Recorded submissions in ${LOG}"
