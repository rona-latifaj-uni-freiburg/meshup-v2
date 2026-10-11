#!/bin/bash
set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

OUTPUT_ROOT=${1:-jobs_with_target_guidance/cross_animal_spike_runs/outputs/densecorr3d_partfield_repair_known6_bad_4000}
EVAL_ROOT=${EVAL_ROOT:-jobs_with_target_guidance/cross_animal_spike_runs/evaluation}
PATHS_FILE=${PATHS_FILE:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_repair_known6_bad_paths.txt}
GEODESIC_PREFIX=${GEODESIC_PREFIX:-densecorr3d_partfield_repair_known6_bad_kim_graph_rawgroups}
SURFACE_DIR=${SURFACE_DIR:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_repair_known6_bad_surface_5k}
SUMMARY_MD=${SUMMARY_MD:-${EVAL_ROOT}/deformation_quality/densecorr3d_partfield_repair_known6_bad_best_summary.md}

python - "${OUTPUT_ROOT}" "${PATHS_FILE}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths_file = Path(sys.argv[2])
paths = sorted(root.glob("*/mesh_final/mesh.obj"))
paths_file.parent.mkdir(parents=True, exist_ok=True)
paths_file.write_text("".join(f"{path}\n" for path in paths))
print(f"found_final_meshes={len(paths)}")
print(f"paths_file={paths_file}")
if not paths:
    raise SystemExit("No repair final meshes found yet.")
PY

source ./activate_meshup_new.sh

python jobs_with_target_guidance/evaluate_densecorr_paper_style_geodesic.py \
  --paths-file "${PATHS_FILE}" \
  --output-dir "${EVAL_ROOT}/bucket_discrepancy" \
  --family-regex "densecorr3d_partfield_repair.*" \
  --ground-truth-label-source densecorr3d_raw \
  --geodesic-method graph \
  --output-prefix "${GEODESIC_PREFIX}"

python jobs_with_target_guidance/evaluate_deformation_quality.py \
  --paths-file "${PATHS_FILE}" \
  --samples 5000 \
  --thresholds 0.01 0.02 0.05 \
  --primary-threshold 0.02 \
  --output-dir "${SURFACE_DIR}"

python - "${EVAL_ROOT}/bucket_discrepancy/${GEODESIC_PREFIX}.csv" "${SURFACE_DIR}/summary.csv" "${SUMMARY_MD}" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

geo_path = Path(sys.argv[1])
surface_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])

surface_by_run = {}
with surface_path.open() as handle:
    for row in csv.DictReader(handle):
        surface_by_run[row["run_name"]] = row

rows = []
with geo_path.open() as handle:
    for row in csv.DictReader(handle):
        surface = surface_by_run.get(row["run_name"], {})
        preset = "unknown"
        for candidate in ("balanced_globalguard", "hybrid_globalguard", "globalguard"):
            if candidate in row["run_name"]:
                preset = candidate
                break
        bucket_match = re.search(r"_pf(\d\d)_", row["run_name"])
        rows.append(
            {
                "pair": row["pair"],
                "preset": preset,
                "pf": f"PF{bucket_match.group(1)}" if bucket_match else "",
                "auc": float(row["paper_style_auc_0_to_0p25"]),
                "err": float(row["paper_style_err_x100"]),
                "same": float(row["nearest_target_same_bucket_rate"]),
                "global_cd": float(surface.get("global_chamfer_l1_sum", "nan")),
                "fscore": float(surface.get("global_fscore", "nan")),
                "normal": float(surface.get("global_normal_consistency", "nan")),
                "output_dir": surface.get("output_dir", ""),
            }
        )

best = {}
for row in rows:
    if row["pair"] not in best or row["err"] < best[row["pair"]]["err"]:
        best[row["pair"]] = row

def mean(items, key):
    vals = [row[key] for row in items]
    return statistics.mean(vals) if vals else float("nan")

def fmt(value):
    return f"{value:.4f}"

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w") as handle:
    handle.write("# DenseCorr3D PartField Repair Summary\n\n")
    handle.write(f"Repair rows evaluated: {len(rows)}. Best-per-pair rows: {len(best)}.\n\n")
    handle.write("## Preset Means\n\n")
    handle.write("| Preset | n | mean AUC | mean Err | same-bucket | global CD | F@0.02 |\n")
    handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for preset in sorted({row["preset"] for row in rows}):
        group = [row for row in rows if row["preset"] == preset]
        handle.write(
            f"| {preset} | {len(group)} | {fmt(mean(group, 'auc'))} | {fmt(mean(group, 'err'))} | "
            f"{fmt(mean(group, 'same'))} | {fmt(mean(group, 'global_cd'))} | {fmt(mean(group, 'fscore'))} |\n"
        )
    best_rows = sorted(best.values(), key=lambda row: row["err"])
    handle.write("\n## Best Repair Per Pair\n\n")
    handle.write("| Pair | preset | PF | AUC | Err | same-bucket | global CD | F@0.02 | output dir |\n")
    handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in best_rows:
        handle.write(
            f"| {row['pair']} | {row['preset']} | {row['pf']} | {fmt(row['auc'])} | {fmt(row['err'])} | "
            f"{fmt(row['same'])} | {fmt(row['global_cd'])} | {fmt(row['fscore'])} | `{row['output_dir']}` |\n"
        )
print(summary_path)
PY
