#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

OUTPUT_ROOT=${1:-jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_partfield_curriculum_try_4000}
BASE_DIR=${BASE_DIR:-jobs_with_target_guidance/cross_animal_spike_runs}
EVAL_ROOT=${EVAL_ROOT:-${BASE_DIR}/evaluation}
SPEC_FILE=${SPEC_FILE:-${BASE_DIR}/jobs/densecorr3d_partfield_curriculum_try_specs.tsv}
PATHS_FILE=${PATHS_FILE:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_curriculum_try_stage3_paths.txt}
GEODESIC_PREFIX=${GEODESIC_PREFIX:-densecorr3d_partfield_curriculum_try_kim_graph_rawgroups}
SURFACE_DIR=${SURFACE_DIR:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_curriculum_try_surface_5k}
SUMMARY_CSV=${SUMMARY_CSV:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_curriculum_try_improvement.csv}
SUMMARY_MD=${SUMMARY_MD:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_curriculum_try_improvement.md}

python - "${OUTPUT_ROOT}" "${PATHS_FILE}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths_file = Path(sys.argv[2])
paths = sorted(root.glob("*stage3*/mesh_final/mesh.obj"))
paths_file.parent.mkdir(parents=True, exist_ok=True)
paths_file.write_text("".join(f"{path}\n" for path in paths))
print(f"found_stage3_final_meshes={len(paths)}")
print(f"paths_file={paths_file}")
if not paths:
    raise SystemExit("No curriculum stage3 final meshes found yet.")
PY

source ./activate_meshup_new.sh

python jobs_with_target_guidance/evaluate_densecorr_paper_style_geodesic.py \
  --paths-file "${PATHS_FILE}" \
  --output-dir "${EVAL_ROOT}/bucket_discrepancy" \
  --family-regex "densecorr3d_partfield_curriculum.*" \
  --ground-truth-label-source densecorr3d_raw \
  --geodesic-method graph \
  --output-prefix "${GEODESIC_PREFIX}"

python jobs_with_target_guidance/evaluate_deformation_quality.py \
  --paths-file "${PATHS_FILE}" \
  --samples 5000 \
  --thresholds 0.01 0.02 0.05 \
  --primary-threshold 0.02 \
  --output-dir "${SURFACE_DIR}"

python - "${SPEC_FILE}" "${EVAL_ROOT}/bucket_discrepancy/${GEODESIC_PREFIX}.csv" "${SURFACE_DIR}/summary.csv" "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

spec_path = Path(sys.argv[1])
geo_path = Path(sys.argv[2])
surface_path = Path(sys.argv[3])
summary_csv = Path(sys.argv[4])
summary_md = Path(sys.argv[5])

baseline = {}
with spec_path.open() as handle:
    reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t",
                            fieldnames=[
                                "pair", "source_mesh", "target_mesh", "source_name", "target_name",
                                "coarse_bucket", "fine_bucket", "source_coarse_labels", "target_coarse_labels",
                                "source_fine_labels", "target_fine_labels", "source_features", "target_features",
                                "prompt", "baseline_pf", "baseline_auc", "baseline_err",
                            ])
    for row in reader:
        baseline[row["pair"]] = row

surface_by_run = {}
with surface_path.open() as handle:
    for row in csv.DictReader(handle):
        surface_by_run[row["run_name"]] = row

rows = []
with geo_path.open() as handle:
    for row in csv.DictReader(handle):
        pair = row["pair"]
        base = baseline[pair]
        surf = surface_by_run.get(row["run_name"], {})
        auc = float(row["paper_style_auc_0_to_0p25"])
        err = float(row["paper_style_err_x100"])
        base_auc = float(base["baseline_auc"])
        base_err = float(base["baseline_err"])
        rows.append(
            {
                "pair": pair,
                "baseline_pf": base["baseline_pf"],
                "coarse_pf": f"PF{base['coarse_bucket']}",
                "fine_pf": f"PF{base['fine_bucket']}",
                "baseline_auc": base_auc,
                "curriculum_auc": auc,
                "delta_auc": auc - base_auc,
                "baseline_err": base_err,
                "curriculum_err": err,
                "delta_err": err - base_err,
                "err_improved": err < base_err,
                "auc_improved": auc > base_auc,
                "same_bucket_rate": float(row["nearest_target_same_bucket_rate"]),
                "global_cd": float(surf.get("global_chamfer_l1_sum", "nan")),
                "fscore_0p02": float(surf.get("global_fscore", "nan")),
                "normal_consistency": float(surf.get("global_normal_consistency", "nan")),
                "output_dir": surf.get("output_dir", ""),
                "final_mesh": row["deformed_mesh"],
            }
        )

rows.sort(key=lambda item: item["pair"])
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

def mean(key):
    return statistics.mean(float(row[key]) for row in rows)

def fmt(value):
    return f"{value:.4f}"

summary_md.parent.mkdir(parents=True, exist_ok=True)
with summary_md.open("w") as handle:
    handle.write("# DenseCorr3D PartField Curriculum Improvement Test\n\n")
    handle.write(f"Evaluated staged final meshes: {len(rows)}.\n\n")
    handle.write("## Mean Change\n\n")
    handle.write("| n | baseline AUC | curriculum AUC | delta AUC | baseline Err | curriculum Err | delta Err | Err improved | AUC improved |\n")
    handle.write("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    handle.write(
        f"| {len(rows)} | {fmt(mean('baseline_auc'))} | {fmt(mean('curriculum_auc'))} | {fmt(mean('delta_auc'))} | "
        f"{fmt(mean('baseline_err'))} | {fmt(mean('curriculum_err'))} | {fmt(mean('delta_err'))} | "
        f"{sum(row['err_improved'] for row in rows)}/{len(rows)} | {sum(row['auc_improved'] for row in rows)}/{len(rows)} |\n\n"
    )
    handle.write("Negative `delta Err` is good. Positive `delta AUC` is good.\n\n")
    handle.write("## Per Pair\n\n")
    handle.write("| Pair | baseline PF | coarse PF | fine PF | baseline AUC | curriculum AUC | delta AUC | baseline Err | curriculum Err | delta Err | global CD | F@0.02 | output dir |\n")
    handle.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in rows:
        handle.write(
            f"| {row['pair']} | {row['baseline_pf']} | {row['coarse_pf']} | {row['fine_pf']} | "
            f"{fmt(row['baseline_auc'])} | {fmt(row['curriculum_auc'])} | {fmt(row['delta_auc'])} | "
            f"{fmt(row['baseline_err'])} | {fmt(row['curriculum_err'])} | {fmt(row['delta_err'])} | "
            f"{fmt(row['global_cd'])} | {fmt(row['fscore_0p02'])} | `{row['output_dir']}` |\n"
        )
print(summary_csv)
print(summary_md)
PY
