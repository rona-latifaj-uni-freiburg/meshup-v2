import csv
import glob
import os
from pathlib import Path

import pymeshlab

INPUT_GLOB = "jobs_with_sam3D/meshes/5k_upright_wheels_down/*.ply"
OUT_DIR = Path("jobs_with_sam3D/meshes/5k_repaired")
SUMMARY = OUT_DIR / "summary.csv"
TARGET_VERTS = 5000


def decimate_to_target(ms: pymeshlab.MeshSet, target_verts: int = TARGET_VERTS):
    target_faces = 10000
    best = None
    for _ in range(6):
        ms_local = pymeshlab.MeshSet()
        ms_local.add_mesh(ms.current_mesh())
        ms_local.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            qualitythr=0.5,
            planarquadric=True,
        )
        m = ms_local.current_mesh()
        v = int(m.vertex_number())
        f = int(m.face_number())
        err = abs(v - target_verts)
        if best is None or err < best[0]:
            best = (err, target_faces, v, f, ms_local)
        if err <= 120:
            break
        ratio = target_verts / max(v, 1)
        next_tf = int(target_faces * ratio)
        target_faces = max(5000, min(30000, next_tf if next_tf != target_faces else target_faces + (500 if v < target_verts else -500)))
    return best


def repair_mesh(src_path: str):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src_path)

    # 1) Weld very close vertices to stitch tiny gaps at component seams.
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.6))

    # 2) Repair non-manifold issues before volumetric reconstruction.
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)

    # 3) Volumetric resampling tends to produce a single connected shell.
    ms.generate_resampled_uniform_mesh(
        cellsize=pymeshlab.PercentageValue(2.0),
        mergeclosevert=True,
        discretize=False,
        multisample=True,
        absdist=False,
    )

    # 4) Light smoothing to stabilize thin noisy spikes.
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=2)

    return ms


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for src in sorted(glob.glob(INPUT_GLOB)):
        if src.endswith("summary.csv"):
            continue
        name = os.path.basename(src)
        stem = os.path.splitext(name)[0]

        repaired = repair_mesh(src)
        err, target_faces, v, f, decimated_ms = decimate_to_target(repaired)

        out_path = OUT_DIR / f"{stem.replace('_5k_upright_wheels_down','')}_5k_repaired.ply"
        decimated_ms.save_current_mesh(str(out_path))

        rows.append({
            "source": name,
            "output": out_path.name,
            "verts": v,
            "faces": f,
            "target_faces_used": target_faces,
            "abs_vert_error": err,
        })

    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "output", "verts", "faces", "target_faces_used", "abs_vert_error"])
        w.writeheader()
        w.writerows(rows)

    print(f"processed={len(rows)}")
    for r in rows:
        print(f"{r['source']}\tverts={r['verts']}\tfaces={r['faces']}\terr={r['abs_vert_error']}\t-> {r['output']}")
    print(f"summary_csv={SUMMARY}")


if __name__ == "__main__":
    main()
