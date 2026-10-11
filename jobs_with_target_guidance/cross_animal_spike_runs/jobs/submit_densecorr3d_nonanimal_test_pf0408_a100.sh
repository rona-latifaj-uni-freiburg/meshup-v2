#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=jobs_with_target_guidance/cross_animal_spike_runs

python "${BASE_DIR}/jobs/generate_densecorr3d_category_partfield_specs.py"

prep_job=$(sbatch --parsable "${BASE_DIR}/jobs/job_gpu_a100_short_densecorr3d_nonanimal_partfield_assets_array.sh")

# 2960 spec rows exceed this cluster's MaxArraySize (1001), so the deform
# sweep is split into three chained sub-arrays (0-999, 1000-1999, 2000-2959)
# via TASK_OFFSET, run one after another to keep total concurrency capped
# at 20 (same cap the animal PF0408 array used).
deform_job1=$(sbatch --parsable --dependency=afterok:${prep_job} --array=0-999%20 --export=ALL,TASK_OFFSET=0 "${BASE_DIR}/jobs/job_gpu_a100_short_densecorr3d_nonanimal_partfield_fair_pf0408_array.sh")
deform_job2=$(sbatch --parsable --dependency=afterany:${deform_job1} --array=0-999%20 --export=ALL,TASK_OFFSET=1000 "${BASE_DIR}/jobs/job_gpu_a100_short_densecorr3d_nonanimal_partfield_fair_pf0408_array.sh")
deform_job3=$(sbatch --parsable --dependency=afterany:${deform_job2} --array=0-959%20 --export=ALL,TASK_OFFSET=2000 "${BASE_DIR}/jobs/job_gpu_a100_short_densecorr3d_nonanimal_partfield_fair_pf0408_array.sh")
eval_job=$(sbatch --parsable --dependency=afterany:${deform_job1}:${deform_job2}:${deform_job3} "${BASE_DIR}/jobs/evaluate_densecorr3d_nonanimal_partfield_fair_pf0408_a100.sh")

echo "prep_job=${prep_job}"
echo "deform_job1(0-999)=${deform_job1}"
echo "deform_job2(1000-1999)=${deform_job2}"
echo "deform_job3(2000-2959)=${deform_job3}"
echo "eval_job=${eval_job}"
echo "outputs=${BASE_DIR}/outputs/densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100"
echo "summary=${BASE_DIR}/evaluation/bucket_discrepancy/densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_a100_kim_graph_rawgroups_bestpf_summary.md"
