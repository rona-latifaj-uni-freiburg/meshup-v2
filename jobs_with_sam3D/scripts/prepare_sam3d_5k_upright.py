import argparse
import csv
import glob
import os
from pathlib import Path

import pymeshlab

TARGET_VERTS = 5000
INPUT_GLOB = "meshes_from_SAM3D/*.ply"
OUT_DIR = Path("jobs_with_sam3D/meshes/5k_upright_wheels_down")
TMP_ORIENTED_DIR = Path("jobs_with_sam3D/meshes/tmp_oriented")
SUMMARY_CSV = OUT_DIR / "summary.csv"
OUTPUT_SUFFIX = "_5k_upright_wheels_down"


def orient_mesh(src_path: str, oriented_path: str) -> None:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(src_path)
    # Match the validated correction used earlier: make car stand correctly and wheels down.
    ms.compute_matrix_from_rotation(rotaxis=0, angle=90.0, freeze=True)
    ms.compute_matrix_from_rotation(rotaxis=2, angle=180.0, freeze=True)
    ms.save_current_mesh(oriented_path)


def decimate_to_target(oriented_path: str, out_path: str, target_verts: int = TARGET_VERTS):
    target_faces = 10000
    best = None

    for _ in range(6):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(oriented_path)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            qualitythr=0.5,
            planarquadric=True,
        )

        cur = ms.current_mesh()
        v = int(cur.vertex_number())
        f = int(cur.face_number())
        err = abs(v - target_verts)

        if best is None or err < best["err"]:
            ms.save_current_mesh(out_path)
            best = {"verts": v, "faces": f, "target_faces": int(target_faces), "err": int(err)}

        if err <= 100:
            break

        ratio = target_verts / max(v, 1)
        next_target_faces = int(target_faces * ratio)
        next_target_faces = max(3000, min(30000, next_target_faces))

        if next_target_faces == target_faces:
            next_target_faces = target_faces + (500 if v < target_verts else -500)

        target_faces = max(3000, min(30000, next_target_faces))

    return best


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orient SAM3D PLY meshes wheels-down and decimate them toward 5k vertices."
    )
    parser.add_argument("--input-glob", default=INPUT_GLOB)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--tmp-oriented-dir", type=Path, default=TMP_ORIENTED_DIR)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--target-verts", type=int, default=TARGET_VERTS)
    parser.add_argument("--output-suffix", default=OUTPUT_SUFFIX)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    tmp_oriented_dir = args.tmp_oriented_dir
    summary_csv = args.summary_csv or out_dir / "summary.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_oriented_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sorted(glob.glob(args.input_glob)):
        name = os.path.basename(src)
        stem = os.path.splitext(name)[0]

        oriented_path = tmp_oriented_dir / f"{stem}_oriented.ply"
        out_path = out_dir / f"{stem}{args.output_suffix}.ply"

        orient_mesh(src, str(oriented_path))
        stats = decimate_to_target(str(oriented_path), str(out_path), args.target_verts)

        rows.append(
            {
                "source": name,
                "output": out_path.name,
                "verts": stats["verts"],
                "faces": stats["faces"],
                "target_faces_used": stats["target_faces"],
                "abs_vert_error": stats["err"],
            }
        )

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "output", "verts", "faces", "target_faces_used", "abs_vert_error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"processed={len(rows)}")
    for r in rows:
        print(
            f"{r['source']}\tverts={r['verts']}\tfaces={r['faces']}\terr={r['abs_vert_error']}\t-> {r['output']}"
        )
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
