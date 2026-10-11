#!/bin/bash
#SBATCH --job-name=eval_dc3d_nonanim
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=jobs_with_target_guidance/cross_animal_spike_runs/logs/eval_densecorr3d_nonanimal_pf0408_a100_%j.out
#SBATCH --error=jobs_with_target_guidance/cross_animal_spike_runs/logs/eval_densecorr3d_nonanimal_pf0408_a100_%j.err

set -euo pipefail

cd /pfs/work9/workspace/scratch/fr_rl187-my_project_ws/projects/meshup_v2

BASE_DIR=${BASE_DIR:-jobs_with_target_guidance/cross_animal_spike_runs}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100}
EVAL_ROOT=${EVAL_ROOT:-${BASE_DIR}/evaluation}
PATHS_FILE=${PATHS_FILE:-${EVAL_ROOT}/deformation_quality/densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_a100_paths.txt}
GEODESIC_PREFIX=${GEODESIC_PREFIX:-densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_a100_kim_graph_rawgroups}
SUMMARY_CSV=${SUMMARY_CSV:-${EVAL_ROOT}/bucket_discrepancy/${GEODESIC_PREFIX}_bestpf_summary.csv}
SUMMARY_MD=${SUMMARY_MD:-${EVAL_ROOT}/bucket_discrepancy/${GEODESIC_PREFIX}_bestpf_summary.md}

mkdir -p "${EVAL_ROOT}/deformation_quality" "${EVAL_ROOT}/bucket_discrepancy"

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
if len(paths) < 2960:
    print(f"WARNING expected 2960 final meshes for non-animal test split category-local pairs x PF4-PF8, found {len(paths)}")
if not paths:
    raise SystemExit("No final meshes found.")
PY

source ./activate_meshup_new.sh

python jobs_with_target_guidance/evaluate_densecorr_paper_style_geodesic.py \
  --paths-file "${PATHS_FILE}" \
  --output-dir "${EVAL_ROOT}/bucket_discrepancy" \
  --family-regex "densecorr3d_nonanimals_test_partfield_guided_fair_pf0408_4000_a100" \
  --ground-truth-label-source densecorr3d_raw \
  --densecorr3d-category auto \
  --geodesic-method graph \
  --output-prefix "${GEODESIC_PREFIX}"

python - "${EVAL_ROOT}/bucket_discrepancy/${GEODESIC_PREFIX}.csv" "${SUMMARY_CSV}" "${SUMMARY_MD}" <<'PY'
import csv
import re
import statistics
import sys
from pathlib import Path

geo_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
summary_md = Path(sys.argv[3])

rows = []
with geo_csv.open() as handle:
    for row in csv.DictReader(handle):
        match = re.search(r"_pf(\d\d)_", row["run_name"])
        pf = f"PF{match.group(1)}" if match else "unknown"
        category = row["target_mesh"].split("/")[2] if row["target_mesh"].startswith("external/DenseCorr3D/") else "unknown"
        rows.append(
            {
                "category": category,
                "pair": row["pair"],
                "pf": pf,
                "auc": float(row["paper_style_auc_0_to_0p25"]),
                "err": float(row["paper_style_err_x100"]),
                "p50_err": float(row["paper_style_p50_x100"]),
                "p95_err": float(row["paper_style_p95_x100"]),
                "same_bucket_rate": float(row["nearest_target_same_bucket_rate"]),
                "target_vertices": int(row["target_vertices"]),
                "deformed_mesh": row["deformed_mesh"],
            }
        )

if not rows:
    raise SystemExit("No rows in geodesic CSV.")

best_by_pair = {}
for row in rows:
    current = best_by_pair.get(row["pair"])
    if current is None or (row["err"], -row["auc"]) < (current["err"], -current["auc"]):
        best_by_pair[row["pair"]] = row

def mean(group, key):
    return statistics.mean(float(row[key]) for row in group) if group else float("nan")

def fmt(value):
    return f"{value:.4f}"

summary_rows = []
for label in sorted({row["pf"] for row in rows}):
    group = [row for row in rows if row["pf"] == label]
    summary_rows.append({
        "setting": label,
        "category": "all",
        "n": len(group),
        "mean_auc": mean(group, "auc"),
        "mean_err": mean(group, "err"),
        "mean_p95_err": mean(group, "p95_err"),
        "mean_same_bucket_rate": mean(group, "same_bucket_rate"),
    })

best_rows = sorted(best_by_pair.values(), key=lambda row: (row["category"], row["pair"]))
summary_rows.append({
    "setting": "Best PF per directed pair",
    "category": "all",
    "n": len(best_rows),
    "mean_auc": mean(best_rows, "auc"),
    "mean_err": mean(best_rows, "err"),
    "mean_p95_err": mean(best_rows, "p95_err"),
    "mean_same_bucket_rate": mean(best_rows, "same_bucket_rate"),
})

for category in sorted({row["category"] for row in best_rows}):
    group = [row for row in best_rows if row["category"] == category]
    summary_rows.append({
        "setting": "Best PF per directed pair",
        "category": category,
        "n": len(group),
        "mean_auc": mean(group, "auc"),
        "mean_err": mean(group, "err"),
        "mean_p95_err": mean(group, "p95_err"),
        "mean_same_bucket_rate": mean(group, "same_bucket_rate"),
    })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["setting", "category", "n", "mean_auc", "mean_err", "mean_p95_err", "mean_same_bucket_rate"],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

with summary_md.open("w") as handle:
    handle.write("# DenseCorr3D Non-Animal Test PartField Fair PF4-PF8 Evaluation\n\n")
    handle.write("Evaluation uses DenseCorr3D raw `groups.txt` labels only for scoring. Deformations are driven by PartField PF4-PF8 buckets.\n\n")
    handle.write("AUC is integrated over normalized graph-geodesic semantic error from 0 to 0.25. Err is mean normalized graph-geodesic semantic error x100.\n\n")
    handle.write("## Summary\n\n")
    handle.write("| Setting | Category | n | mean AUC | mean Err | mean p95 Err | mean same-bucket rate |\n")
    handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
    for row in summary_rows:
        handle.write(
            f"| {row['setting']} | {row['category']} | {row['n']} | {fmt(row['mean_auc'])} | "
            f"{fmt(row['mean_err'])} | {fmt(row['mean_p95_err'])} | {fmt(row['mean_same_bucket_rate'])} |\n"
        )
    handle.write("\n## Best PF Per Directed Pair\n\n")
    handle.write("| Category | Pair | best PF | AUC | Err | p50 Err | p95 Err | same-bucket | target verts | deformed mesh |\n")
    handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for row in best_rows:
        handle.write(
            f"| {row['category']} | `{row['pair']}` | {row['pf']} | {fmt(row['auc'])} | {fmt(row['err'])} | "
            f"{fmt(row['p50_err'])} | {fmt(row['p95_err'])} | {fmt(row['same_bucket_rate'])} | "
            f"{row['target_vertices']} | `{row['deformed_mesh']}` |\n"
        )

print(summary_csv)
print(summary_md)
print(f"all_rows={len(rows)}")
print(f"best_pairs={len(best_rows)}")
PY
